/* כלי הגרף לבדיקת אסטרטגיות (Strategy Lab):
   - עסקאות כקווים (כניסה→יציאה), סטופ/יעד, גיאומטריה של האסטרטגיה (רגל פיבונאצ'י, טווח וויקוף, טווח פתיחה)
   - ניווט קודם/הבא בין עסקאות, קפיצה מטבלת העסקאות, הדגשת העסקה הנבחרת, ריחוף עם פרטים
   - חלון נתונים (OHLCV, שינוי, שעה בישראל ובניו-יורק)
   - הצללת הסשן הרגיל של ארה"ב, טווח הבדיקה וחלונות Walk-forward
   - רצועת עקומת הון מסונכרנת לזמן
   - סרגל כלים: איפוס זום, קפיצה לעכשיו, מסך מלא; קו אופקי / קו מגמה עם שמירה
   - מסחר בנייר: כניסות בצבע אחר וקו הפוזיציה הפתוחה
   מתחבר ל-window.__tv שחושף chart.js (הקוד המקורי לא משתנה). */
(function () {
    "use strict";
    var S = {
        trades: [], byEntry: {}, index: [], selIdx: -1, selEntry: null,
        ctx: {tested: null, wf: [], equity: [], paper: {trades: [], open: []}},
        drawings: [], tool: null, pending: null, hoverTrade: null, selDraw: -1,
        showEq: true, showSess: true, showCtx: true,
        overlay: null, eqChart: null, eqSeries: null, timesCache: [], lastDataLen: -1,
        seq: 0, hitboxes: [], symbolKey: "",
    };
    var C = {
        long: "#1FBF8F", short: "#F0475A", entry: "#C77DFF", sel: "#FFFFFF", sl: "#F0475A", tp: "#1FBF8F",
        geo: "rgba(255,196,64,.85)", geoFill: "rgba(255,196,64,.08)", sess: "rgba(61,123,255,.06)",
        tested: "rgba(31,191,143,.05)", train: "rgba(139,144,160,.07)", test: "rgba(61,123,255,.10)",
        paper: "#22D3EE", draw: "#F5B84B",
    };
    function tv() { return window.__tv; }
    function $(id) { return document.getElementById(id); }
    function symbol() { var el = $("symbol-input"); return el ? (el.value || "").trim().toUpperCase() : ""; }
    function tf() { var b = document.querySelector(".tf-btn.active"); return b ? b.textContent.trim() : ""; }
    function fmtMoney(v) { var s = (v < 0 ? "-" : "") + "$" + Math.abs(v).toLocaleString("en-US", {maximumFractionDigits: 0}); return s; }
    function fmtDur(m) { if (m < 90) return Math.round(m) + " דק'"; if (m < 2880) return (m / 60).toFixed(1) + " שע'"; return (m / 1440).toFixed(1) + " ימים"; }
    function tzOffsetMin(tz, date) {
        var f = new Intl.DateTimeFormat("en-US", {timeZone: tz, hour12: false, year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit"});
        var p = {}; f.formatToParts(date).forEach(function (x) { p[x.type] = x.value; });
        var asUTC = Date.UTC(+p.year, +p.month - 1, +p.day, (+p.hour) % 24, +p.minute);
        return (asUTC - date.getTime()) / 60000;
    }
    function fmtTime(epoch, tz) {
        // ה-epoch הוא "UTC מזויף": רכיבי UTC שלו הם שעון ישראל
        var d = new Date(epoch * 1000);
        var pad = function (n) { return (n < 10 ? "0" : "") + n; };
        if (tz === "il") return pad(d.getUTCDate()) + "/" + pad(d.getUTCMonth() + 1) + " " + pad(d.getUTCHours()) + ":" + pad(d.getUTCMinutes());
        try {
            var instant = new Date(d.getTime() - tzOffsetMin("Asia/Jerusalem", d) * 60000);
            var f = new Intl.DateTimeFormat("en-GB", {timeZone: "America/New_York", hour12: false, day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit"});
            var p = {}; f.formatToParts(instant).forEach(function (x) { p[x.type] = x.value; });
            return p.day + "/" + p.month + " " + p.hour + ":" + p.minute;
        } catch (e) { return ""; }
    }
    function tfSeconds() { var m = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400, "1w": 604800}; return m[tf()] || 300; }

    // ---------- מציאת קואורדינטת זמן גם כשהזמן המדויק לא קיים בסדרה (LOD גס) ----------
    function refreshTimes() {
        var t = tv(); if (!t || !t.candleSeries || typeof t.candleSeries.data !== "function") return;
        var data = t.candleSeries.data();
        if (data.length !== S.lastDataLen || (data.length && S.timesCache[0] !== data[0].time)) {
            S.timesCache = data.map(function (c) { return c.time; });
            S.lastDataLen = data.length;
        }
    }
    function nearestTime(epoch) {
        var a = S.timesCache; if (!a.length) return null;
        var lo = 0, hi = a.length - 1;
        if (epoch <= a[0]) return a[0]; if (epoch >= a[hi]) return a[hi];
        while (hi - lo > 1) { var mid = (lo + hi) >> 1; if (a[mid] <= epoch) lo = mid; else hi = mid; }
        return (epoch - a[lo] <= a[hi] - epoch) ? a[lo] : a[hi];
    }
    function xOf(epoch, ts) {
        var x = ts.timeToCoordinate(epoch);
        if (x !== null) return x;
        var nt = nearestTime(epoch); if (nt === null) return null;
        x = ts.timeToCoordinate(nt);
        if (x === null) return null;
        // התאמה ליניארית בתוך המרווח בין נרות (כשהזמן נופל בין שני נרות)
        var a = S.timesCache, i = a.indexOf(nt);
        var nb = (epoch > nt) ? a[i + 1] : a[i - 1];
        if (nb === undefined) return x;
        var xb = ts.timeToCoordinate(nb); if (xb === null) return x;
        return x + (xb - x) * (epoch - nt) / (nb - nt);
    }

    // ---------- הפרימיטיב: שכבת ציור מתחת לנרות (רקע) ומעליהם (עסקאות) ----------
    function createOverlay() {
        var chartRef = null, seriesRef = null, requestUpdate = null;
        function drawBottom(scope) {
            var ctx = scope.context, ts = chartRef.timeScale(), W = scope.mediaSize.width, H = scope.mediaSize.height;
            refreshTimes();
            var vr = ts.getVisibleRange(); if (!vr) return;
            // הצללת הסשן הרגיל (16:30–23:00 שעון ישראל = 09:30–16:00 ניו-יורק)
            if (S.showSess && S.timesCache.length) {
                ctx.fillStyle = C.sess;
                var dayStart = Math.floor(vr.from / 86400) * 86400 - 86400;
                for (var d = dayStart; d <= vr.to + 86400; d += 86400) {
                    var a = d + 16 * 3600 + 1800, b = d + 23 * 3600;
                    if (b < vr.from || a > vr.to) continue;
                    var x1 = xOf(Math.max(a, vr.from), ts), x2 = xOf(Math.min(b, vr.to), ts);
                    if (x1 === null || x2 === null) continue;
                    ctx.fillRect(Math.min(x1, x2), 0, Math.abs(x2 - x1), H);
                }
            }
            if (S.showCtx) {
                if (S.ctx.tested) {
                    var t1 = xOf(Math.max(S.ctx.tested.from, vr.from), ts), t2 = xOf(Math.min(S.ctx.tested.to, vr.to), ts);
                    if (t1 !== null && t2 !== null && S.ctx.tested.from <= vr.to && S.ctx.tested.to >= vr.from) {
                        ctx.fillStyle = C.tested; ctx.fillRect(Math.min(t1, t2), 0, Math.abs(t2 - t1), H);
                    }
                }
                (S.ctx.wf || []).forEach(function (w) {
                    [["train_from", "train_to", C.train, "אימון"], ["test_from", "test_to", C.test, "בדיקה"]].forEach(function (seg) {
                        var f = w[seg[0]], t = w[seg[1]]; if (t < vr.from || f > vr.to) return;
                        var x1 = xOf(Math.max(f, vr.from), ts), x2 = xOf(Math.min(t, vr.to), ts); if (x1 === null || x2 === null) return;
                        ctx.fillStyle = seg[2]; ctx.fillRect(Math.min(x1, x2), H - 14, Math.abs(x2 - x1), 14);
                        ctx.fillStyle = "rgba(230,232,238,.7)"; ctx.font = "10px Heebo, sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
                        if (Math.abs(x2 - x1) > 40) ctx.fillText(seg[3], (x1 + x2) / 2, H - 7);
                    });
                });
            }
        }
        function drawTop(scope) {
            var ctx = scope.context, ts = chartRef.timeScale(), W = scope.mediaSize.width;
            refreshTimes();
            S.hitboxes = [];
            ctx.save();
            ctx.lineWidth = 1.5; ctx.font = "600 10px Heebo, sans-serif"; ctx.textAlign = "left"; ctx.textBaseline = "middle";
            S.trades.forEach(function (t) {
                var x1 = xOf(t.et, ts), x2 = xOf(t.xt, ts);
                var y1 = seriesRef.priceToCoordinate(t.ep), y2 = seriesRef.priceToCoordinate(t.xp);
                if (x1 === null || x2 === null || y1 === null || y2 === null) return;
                var sel = (t.et === S.selEntry), hov = (S.hoverTrade === t);
                var col = t.pnl >= 0 ? C.long : C.short;
                // גיאומטריה של האסטרטגיה (רק לעסקה הנבחרת/מרוחפת, כדי לא להעמיס)
                if ((sel || hov) && t.extra) {
                    t.extra.forEach(function (g) {
                        var gx1 = xOf(g.x1, ts), gx2 = xOf(g.x2, ts); if (gx1 === null || gx2 === null) return;
                        ctx.strokeStyle = C.geo; ctx.fillStyle = C.geoFill; ctx.setLineDash([4, 3]);
                        if (g.t === "box") {
                            var gy1 = seriesRef.priceToCoordinate(g.y1), gy2 = seriesRef.priceToCoordinate(g.y2); if (gy1 === null || gy2 === null) return;
                            ctx.fillRect(Math.min(gx1, gx2), Math.min(gy1, gy2), Math.abs(gx2 - gx1), Math.abs(gy2 - gy1));
                            ctx.strokeRect(Math.min(gx1, gx2), Math.min(gy1, gy2), Math.abs(gx2 - gx1), Math.abs(gy2 - gy1));
                        } else if (g.t === "seg") {
                            var sy1 = seriesRef.priceToCoordinate(g.y1), sy2 = seriesRef.priceToCoordinate(g.y2); if (sy1 === null || sy2 === null) return;
                            ctx.beginPath(); ctx.moveTo(gx1, sy1); ctx.lineTo(gx2, sy2); ctx.stroke();
                        } else if (g.t === "hline") {
                            var hy = seriesRef.priceToCoordinate(g.y); if (hy === null) return;
                            ctx.beginPath(); ctx.moveTo(gx1, hy); ctx.lineTo(gx2, hy); ctx.stroke();
                        }
                        if (g.label) { ctx.fillStyle = C.geo; ctx.fillText(g.label, Math.min(gx1, gx2) + 4, (g.t === "hline" ? seriesRef.priceToCoordinate(g.y) : seriesRef.priceToCoordinate(g.y2 || g.y1)) - 8); }
                        ctx.setLineDash([]);
                    });
                }
                // סטופ ויעד כקווים מקווקווים לאורך העסקה
                if (t.sl !== undefined || t.tp !== undefined) {
                    ctx.setLineDash([3, 3]); ctx.lineWidth = 1;
                    if (t.sl !== undefined) { var ys = seriesRef.priceToCoordinate(t.sl); if (ys !== null) { ctx.strokeStyle = C.sl; ctx.beginPath(); ctx.moveTo(x1, ys); ctx.lineTo(x2, ys); ctx.stroke(); } }
                    if (t.tp !== undefined) { var yt = seriesRef.priceToCoordinate(t.tp); if (yt !== null) { ctx.strokeStyle = C.tp; ctx.beginPath(); ctx.moveTo(x1, yt); ctx.lineTo(x2, yt); ctx.stroke(); } }
                    ctx.setLineDash([]); ctx.lineWidth = 1.5;
                }
                // קו העסקה + נקודות כניסה/יציאה
                ctx.strokeStyle = sel || hov ? C.sel : col; ctx.lineWidth = sel ? 2.5 : (hov ? 2 : 1.5);
                ctx.globalAlpha = (S.selEntry !== null && !sel && !hov) ? 0.55 : 1;
                ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
                ctx.fillStyle = t.side === "long" ? C.long : C.short;
                ctx.beginPath(); ctx.arc(x1, y1, sel ? 4.5 : 3, 0, Math.PI * 2); ctx.fill();
                ctx.fillStyle = col; ctx.beginPath(); ctx.arc(x2, y2, sel ? 4.5 : 3, 0, Math.PI * 2); ctx.fill();
                ctx.globalAlpha = 1;
                if (sel || hov) { ctx.fillStyle = "#FFFFFF"; ctx.fillText((t.side === "long" ? "LONG " : "SHORT ") + fmtMoney(t.pnl), Math.min(x1, x2) + 6, Math.min(y1, y2) - 10); }
                S.hitboxes.push({t: t, x1: x1, y1: y1, x2: x2, y2: y2});
            });
            // מסחר בנייר
            (S.ctx.paper.trades || []).forEach(function (t) {
                var x1 = xOf(t.et, ts), x2 = xOf(t.xt, ts), y1 = seriesRef.priceToCoordinate(t.ep), y2 = seriesRef.priceToCoordinate(t.xp);
                if (x1 === null || x2 === null || y1 === null || y2 === null) return;
                ctx.strokeStyle = C.paper; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
                ctx.fillStyle = C.paper; ctx.beginPath(); ctx.arc(x1, y1, 3.5, 0, Math.PI * 2); ctx.fill(); ctx.beginPath(); ctx.arc(x2, y2, 3.5, 0, Math.PI * 2); ctx.fill();
                ctx.fillText("נייר " + fmtMoney(t.pnl), Math.min(x1, x2) + 6, Math.min(y1, y2) - 10);
            });
            (S.ctx.paper.open || []).forEach(function (t) {
                var x1 = xOf(t.et, ts), y1 = seriesRef.priceToCoordinate(t.ep); if (x1 === null || y1 === null) return;
                ctx.strokeStyle = C.paper; ctx.setLineDash([6, 4]); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(W, y1); ctx.stroke(); ctx.setLineDash([]);
                ctx.fillStyle = C.paper; ctx.beginPath(); ctx.arc(x1, y1, 4, 0, Math.PI * 2); ctx.fill();
                ctx.fillText("פוזיציה פתוחה בנייר · " + (t.side === "long" ? "לונג" : "שורט") + " · " + fmtMoney(t.unrealized || 0), x1 + 8, y1 - 10);
            });
            // ציורים ידניים
            S.drawings.forEach(function (d, i) {
                var isSel = (i === S.selDraw);
                ctx.strokeStyle = isSel ? "#FFFFFF" : C.draw; ctx.lineWidth = isSel ? 2.5 : 1.5;
                if (d.type === "h") { var y = seriesRef.priceToCoordinate(d.p); if (y === null) return; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
                    ctx.fillStyle = isSel ? "#FFFFFF" : C.draw; ctx.fillText(d.p.toFixed(2), 6, y - 8);
                    if (isSel) { ctx.fillStyle = "#FFFFFF"; ctx.beginPath(); ctx.arc(W / 2, y, 4, 0, Math.PI * 2); ctx.fill(); }
                    d._sx = W / 2; d._sy = y; }
                else if (d.type === "t") { var ax = xOf(d.t1, ts), bx = xOf(d.t2, ts), ay = seriesRef.priceToCoordinate(d.p1), by = seriesRef.priceToCoordinate(d.p2);
                    if (ax === null || bx === null || ay === null || by === null) return;
                    var dx = bx - ax, dy = by - ay, ex = bx, ey = by; if (dx !== 0) { ex = W; ey = by + dy * (W - bx) / dx; }
                    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(ex, ey); ctx.stroke();
                    if (isSel) { ctx.fillStyle = "#FFFFFF"; [[ax, ay], [bx, by]].forEach(function (pt) { ctx.beginPath(); ctx.arc(pt[0], pt[1], 4, 0, Math.PI * 2); ctx.fill(); }); }
                    d._sx = (ax + bx) / 2; d._sy = (ay + by) / 2; }
            });
            positionDeleteButton();
            if (S.pending && S.tool === "t") { var px = xOf(S.pending.t, ts), py = seriesRef.priceToCoordinate(S.pending.p); if (px !== null && py !== null) { ctx.fillStyle = C.draw; ctx.beginPath(); ctx.arc(px, py, 4, 0, Math.PI * 2); ctx.fill(); } }
            ctx.restore();
        }
        var bottomView = {renderer: function () { return {draw: function (target) { target.useMediaCoordinateSpace(function (scope) { if (chartRef && seriesRef) drawBottom(scope); }); }}; }, zOrder: function () { return "bottom"; }};
        var topView = {renderer: function () { return {draw: function (target) { target.useMediaCoordinateSpace(function (scope) { if (chartRef && seriesRef) drawTop(scope); }); }}; }, zOrder: function () { return "top"; }};
        return {
            attached: function (p) { chartRef = p.chart; seriesRef = p.series; requestUpdate = p.requestUpdate; },
            detached: function () { chartRef = seriesRef = requestUpdate = null; },
            paneViews: function () { return [bottomView, topView]; },
            updateAllViews: function () {},
            refresh: function () { if (requestUpdate) requestUpdate(); },
        };
    }

    // ---------- נתונים מהשרת ----------
    function fetchTrades() {
        var t = tv(); if (!t || !t.chart) return;
        var vr = t.chart.timeScale().getVisibleRange(); if (!vr) return;
        var seq = ++S.seq;
        fetch("/api/trades?symbol=" + encodeURIComponent(symbol()) + "&timeframe=" + encodeURIComponent(tf()) +
              "&from=" + Math.round(vr.from) + "&to=" + Math.round(vr.to) + "&seq=" + seq)
            .then(function (r) { return r.json(); }).then(function (p) {
                if (Number(p.seq) !== seq) return;
                S.trades = p.trades || []; S.byEntry = {}; S.trades.forEach(function (x) { S.byEntry[x.et] = x; });
                if (S.overlay) S.overlay.refresh();
            }).catch(function () {});
    }
    function fetchIndexAndContext() {
        fetch("/api/trade_index?symbol=" + encodeURIComponent(symbol()) + "&timeframe=" + encodeURIComponent(tf()))
            .then(function (r) { return r.json(); }).then(function (p) { S.index = p.index || []; S.selIdx = -1; S.selEntry = null; updateCounter(); }).catch(function () {});
        fetch("/api/context?symbol=" + encodeURIComponent(symbol()) + "&timeframe=" + encodeURIComponent(tf()))
            .then(function (r) { return r.json(); }).then(function (p) { S.ctx = p; renderEquity(); if (S.overlay) S.overlay.refresh(); }).catch(function () {});
    }
    function updateCounter() {
        var el = $("ct-counter"); if (!el) return;
        el.textContent = S.index.length ? ((S.selIdx >= 0 ? (S.selIdx + 1) : "—") + " / " + S.index.length.toLocaleString()) : "אין עסקאות";
    }

    // ---------- ניווט ----------
    function jumpTo(epoch, idx) {
        var t = tv(); if (!t || !t.chart) return;
        var ts = t.chart.timeScale(), vr = ts.getVisibleRange();
        var bar = tfSeconds();
        var span = vr ? (vr.to - vr.from) : bar * 200;
        span = Math.max(bar * 60, Math.min(span, bar * 800));   // תמיד רואים הקשר סביב העסקה
        S.selEntry = epoch;
        if (idx === undefined) { idx = S.index.findIndex(function (r) { return r[0] === epoch; }); }
        S.selIdx = idx; updateCounter();
        var from = epoch - span * 0.45, to = epoch + span * 0.55;
        var apply = function () {
            var a = t.closestCandleIndex ? t.closestCandleIndex(from) : null, b = t.closestCandleIndex ? t.closestCandleIndex(to) : null;
            if (a !== null && b !== null && b > a) ts.setVisibleLogicalRange({from: a, to: b});
            else ts.setVisibleRange({from: from, to: to});
            setTimeout(fetchTrades, 250);
            if (S.overlay) S.overlay.refresh();
        };
        // אם היעד מחוץ לדאטה הטעון - טוענים קודם (כמו "Go to date"), אחרת הספרייה מכווצת את הטווח לנר בודד
        if (t.requestLodWindowAsync && t.idealTierFor) {
            t.requestLodWindowAsync(t.idealTierFor(span), from, to, span).then(apply, apply);
        } else { apply(); }
    }
    function step(delta) {
        if (!S.index.length) return;
        var i = S.selIdx < 0 ? (delta > 0 ? 0 : S.index.length - 1) : Math.max(0, Math.min(S.index.length - 1, S.selIdx + delta));
        jumpTo(S.index[i][0], i);
    }

    // ---------- חלון נתונים + ריחוף ----------
    function setupCrosshair() {
        var t = tv(); var win = $("chart-datawin"), tip = $("chart-tooltip"); if (!t || !win) return;
        t.chart.subscribeCrosshairMove(function (param) {
            if (!param || !param.time || !param.seriesData) { win.style.display = "none"; return; }
            var c = param.seriesData.get(t.candleSeries); if (!c) { win.style.display = "none"; return; }
            var v = t.volumeSeries ? param.seriesData.get(t.volumeSeries) : null;
            var chg = c.close - c.open, pct = c.open ? chg / c.open * 100 : 0;
            var cls = chg >= 0 ? "pos" : "neg";
            win.innerHTML =
                '<div class="dw-row"><span>ישראל</span><b>' + fmtTime(param.time, "il") + '</b><span>ניו-יורק</span><b>' + fmtTime(param.time, "ny") + '</b></div>' +
                '<div class="dw-row"><span>O</span><b>' + c.open.toFixed(2) + '</b><span>H</span><b>' + c.high.toFixed(2) + '</b><span>L</span><b>' + c.low.toFixed(2) + '</b><span>C</span><b>' + c.close.toFixed(2) + '</b></div>' +
                '<div class="dw-row"><span>שינוי</span><b class="' + cls + '">' + (chg >= 0 ? "+" : "") + chg.toFixed(2) + " (" + pct.toFixed(2) + "%)</b>" +
                (v && v.value !== undefined ? '<span>נפח</span><b>' + Math.round(v.value).toLocaleString() + '</b>' : "") + '</div>';
            win.style.display = "block";
            // ריחוף על עסקה
            var pt = param.point; var best = null, bestD = 12;
            if (pt) S.hitboxes.forEach(function (h) {
                var d1 = Math.hypot(h.x1 - pt.x, h.y1 - pt.y), d2 = Math.hypot(h.x2 - pt.x, h.y2 - pt.y);
                var d = Math.min(d1, d2); if (d < bestD) { bestD = d; best = h.t; }
            });
            if (best !== S.hoverTrade) { S.hoverTrade = best; if (S.overlay) S.overlay.refresh(); }
            if (best && tip) {
                tip.innerHTML = '<b>' + (best.side === "long" ? "לונג" : "שורט") + '</b> · עסקה #' + (best.i + 1) +
                    '<br>כניסה ' + best.ep.toFixed(2) + ' → יציאה ' + best.xp.toFixed(2) +
                    '<br><span class="' + (best.pnl >= 0 ? "pos" : "neg") + '">' + fmtMoney(best.pnl) + ' (' + best.ret.toFixed(2) + '%)</span> · ' + fmtDur(best.dur) +
                    (best.sl !== undefined ? '<br>סטופ ' + best.sl.toFixed(2) + (best.tp !== undefined ? ' · יעד ' + best.tp.toFixed(2) : "") : "");
                tip.style.display = "block";
                var pane = document.querySelector(".chart-pane"); var pr = pane.getBoundingClientRect();
                var left = pt.x + 14, top = pt.y + 14;
                if (left + 240 > pr.width) left = pt.x - 250; if (top + 90 > pr.height) top = pt.y - 100;
                tip.style.left = left + "px"; tip.style.top = top + "px";
            } else if (tip) { tip.style.display = "none"; }
        });
    }

    // ---------- רצועת עקומת הון ----------
    function renderEquity() {
        var strip = $("equity-strip"), t = tv(); if (!strip || !t) return;
        var eq = S.ctx.equity || [];
        if (!S.showEq || eq.length < 2) { strip.style.display = "none"; document.documentElement.style.setProperty("--eq-h", "0px"); return; }
        strip.style.display = "block"; document.documentElement.style.setProperty("--eq-h", "92px");
        if (!S.eqChart) {
            S.eqChart = LightweightCharts.createChart(strip, {
                width: strip.clientWidth, height: 92, layout: {background: {color: "transparent"}, textColor: "#8B90A0", fontFamily: "Heebo, sans-serif", fontSize: 10},
                grid: {vertLines: {visible: false}, horzLines: {color: "rgba(42,46,55,.5)"}},
                rightPriceScale: {borderVisible: false, scaleMargins: {top: 0.15, bottom: 0.1}}, timeScale: {visible: false},
                crosshair: {horzLine: {visible: false}, vertLine: {visible: false}}, handleScroll: false, handleScale: false,
            });
            S.eqSeries = (typeof S.eqChart.addAreaSeries === "function") ? S.eqChart.addAreaSeries({lineWidth: 1.5, priceLineVisible: false, lastValueVisible: true, crosshairMarkerVisible: false})
                : S.eqChart.addSeries(LightweightCharts.AreaSeries, {lineWidth: 1.5, priceLineVisible: false, lastValueVisible: true});
            new ResizeObserver(function () { if (S.eqChart) S.eqChart.applyOptions({width: strip.clientWidth, height: 92}); syncEquityRange(); }).observe(strip);
        }
        var last = eq[eq.length - 1][1]; var up = last >= 0;
        S.eqSeries.applyOptions({lineColor: up ? "#1FBF8F" : "#F0475A", topColor: up ? "rgba(31,191,143,.25)" : "rgba(240,71,90,.25)", bottomColor: "rgba(0,0,0,0)"});
        // נקודות ייחודיות בזמן (הספרייה דורשת זמנים עולים ממש)
        var pts = [], prev = null; eq.forEach(function (r) { if (prev !== null && r[0] <= prev) return; pts.push({time: r[0], value: r[1]}); prev = r[0]; });
        S.eqSeries.setData(pts);
        syncEquityRange();
    }
    function syncEquityRange() {
        var t = tv(); if (!t || !S.eqChart || !S.eqSeries) return;
        var vr = t.chart.timeScale().getVisibleRange(); if (!vr) return;
        try { S.eqChart.timeScale().setVisibleRange({from: vr.from, to: vr.to}); } catch (e) {}
    }

    // ---------- ציורים ----------
    function drawKey() { return "sl_draw_" + symbol(); }
    function loadDrawings() { try { S.drawings = JSON.parse(localStorage.getItem(drawKey()) || "[]"); } catch (e) { S.drawings = []; } }
    function saveDrawings() { try { localStorage.setItem(drawKey(), JSON.stringify(S.drawings)); } catch (e) {} }
    function setTool(tool) {
        S.tool = (S.tool === tool) ? null : tool; S.pending = null;
        ["ct-hline", "ct-tline"].forEach(function (id) { var b = $(id); if (b) b.classList.toggle("active", (id === "ct-hline" && S.tool === "h") || (id === "ct-tline" && S.tool === "t")); });
        var el = $("chart-container"); if (el) el.style.cursor = S.tool ? "crosshair" : "";
    }
    function chartPointFromEvent(e) {
        var t = tv(), el = $("chart-container"); if (!t || !el) return null;
        var r = el.getBoundingClientRect(); var x = e.clientX - r.left, y = e.clientY - r.top;
        var time = t.chart.timeScale().coordinateToTime(x); var price = t.candleSeries.coordinateToPrice(y);
        if (time === null || price === null) return null;
        return {t: time, p: price, x: x, y: y};
    }
    function onChartClick(e) {
        if (!S.tool) { onChartSelectClick(e); return; }
        var pt = chartPointFromEvent(e); if (!pt) return;
        e.preventDefault(); e.stopPropagation();
        if (S.tool === "h") { S.drawings.push({type: "h", p: pt.p}); saveDrawings(); setTool(null); }
        else if (S.tool === "t") {
            if (!S.pending) { S.pending = pt; }
            else { S.drawings.push({type: "t", t1: S.pending.t, p1: S.pending.p, t2: pt.t, p2: pt.p}); S.pending = null; saveDrawings(); setTool(null); }
        }
        if (S.overlay) S.overlay.refresh();
    }
    function drawingAt(pt) {
        var t = tv(); if (!t) return -1;
        var ts = t.chart.timeScale(); var hit = -1, best = 9;
        S.drawings.forEach(function (d, i) {
            var dist = 999;
            if (d.type === "h") { var y = t.candleSeries.priceToCoordinate(d.p); if (y !== null) dist = Math.abs(y - pt.y); }
            else { var ax = xOf(d.t1, ts), bx = xOf(d.t2, ts), ay = t.candleSeries.priceToCoordinate(d.p1), by = t.candleSeries.priceToCoordinate(d.p2);
                if (ax === null || bx === null || ay === null || by === null) return;
                var dx = bx - ax, dy = by - ay, len = Math.hypot(dx, dy) || 1; dist = Math.abs(dy * pt.x - dx * pt.y + bx * ay - by * ax) / len;
                if (pt.x < Math.min(ax, bx) - 6) dist = 999; }
            if (dist < best) { best = dist; hit = i; }
        });
        return hit;
    }
    function selectDrawing(i) {
        S.selDraw = i;
        if (S.overlay) S.overlay.refresh();
        positionDeleteButton();
    }
    function deleteSelectedDrawing() {
        if (S.selDraw < 0) return;
        S.drawings.splice(S.selDraw, 1); S.selDraw = -1; saveDrawings();
        if (S.overlay) S.overlay.refresh(); positionDeleteButton();
    }
    function positionDeleteButton() {
        var btn = $("draw-del"); if (!btn) return;
        var d = S.drawings[S.selDraw];
        if (!d || d._sx === undefined || d._sy === undefined) { btn.style.display = "none"; return; }
        btn.style.display = "flex"; btn.style.left = Math.max(4, d._sx + 10) + "px"; btn.style.top = Math.max(4, d._sy - 30) + "px";
    }
    function onChartContextMenu(e) {
        var pt = chartPointFromEvent(e); if (!pt) return;
        var hit = drawingAt(pt);
        if (hit >= 0) { e.preventDefault(); e.stopPropagation(); S.selDraw = hit; deleteSelectedDrawing(); }
    }
    function onChartSelectClick(e) {
        if (S.tool) return;
        var pt = chartPointFromEvent(e); if (!pt) return;
        var hit = drawingAt(pt);
        if (hit !== S.selDraw) selectDrawing(hit);
    }

    // ---------- חיבור ----------
    function attach() {
        var t = tv(); if (!t || !t.chart || !t.candleSeries || typeof t.candleSeries.attachPrimitive !== "function") return false;
        if (S.overlay) return true;
        S.overlay = createOverlay(); t.candleSeries.attachPrimitive(S.overlay);
        loadDrawings(); S.symbolKey = symbol();
        var deb = null;
        t.chart.timeScale().subscribeVisibleTimeRangeChange(function () {
            clearTimeout(deb); deb = setTimeout(fetchTrades, 180); syncEquityRange();
            if (S.overlay) S.overlay.refresh();
        });
        setupCrosshair();
        var el = $("chart-container");
        el.addEventListener("click", onChartClick, true);
        el.addEventListener("contextmenu", onChartContextMenu, true);
        el.addEventListener("mousemove", function (e) {
            if (S.tool) return;
            var pt = chartPointFromEvent(e); if (!pt) return;
            el.style.cursor = (drawingAt(pt) >= 0) ? "pointer" : "";
        });
        var del = document.createElement("button");
        del.id = "draw-del"; del.className = "draw-del"; del.textContent = "✕ מחק קו"; del.title = "מחיקת הקו שנבחר (Delete)"; del.style.display = "none";
        del.addEventListener("click", function (ev) { ev.stopPropagation(); deleteSelectedDrawing(); });
        document.querySelector(".chart-pane").appendChild(del);
        fetchIndexAndContext(); fetchTrades();
        return true;
    }
    var poll = setInterval(function () { if (attach()) clearInterval(poll); }, 400);
    // פריסה שמורה (זו-לצד-זו / למעלה-למטה)
    (function () { try { if (localStorage.getItem("sl_layout") === "stacked") { var w = setInterval(function () { var ws = document.querySelector(".workspace"); if (ws) { ws.classList.add("stacked"); clearInterval(w); } }, 200); } } catch (e) {} })();

    // ריצה חדשה (run-label משתנה) → רענון האינדקס, ההקשר והעסקאות; שינוי סימבול → ציורים של הסימבול
    var mo = new MutationObserver(function () { fetchIndexAndContext(); setTimeout(fetchTrades, 300); });
    var arm = setInterval(function () { var lbl = $("run-label"); if (lbl) { mo.observe(lbl, {childList: true, characterData: true, subtree: true}); clearInterval(arm); } }, 500);
    setInterval(function () { if (symbol() !== S.symbolKey) { S.symbolKey = symbol(); loadDrawings(); fetchIndexAndContext(); fetchTrades(); } }, 1500);
    setInterval(function () { if (S.ctx.paper && (S.ctx.paper.open.length || S.ctx.paper.trades.length)) fetchIndexAndContext(); }, 15000);

    // סרגל כלים + מקלדת + לחיצה על שורת עסקה בטבלה
    document.addEventListener("click", function (e) {
        var id = e.target && e.target.id;
        var t = tv();
        if (id === "ct-prev") step(-1);
        else if (id === "ct-next") step(1);
        else if (id === "ct-fit" && t) { t.chart.timeScale().fitContent(); }
        else if (id === "ct-now" && t) { t.chart.timeScale().scrollToRealTime(); }
        else if (id === "ct-full") { var b = $("panel-toggle"); if (b) b.click(); }
        else if (id === "ct-eq") { S.showEq = !S.showEq; e.target.classList.toggle("on", S.showEq); renderEquity(); }
        else if (id === "ct-sess") { S.showSess = !S.showSess; e.target.classList.toggle("on", S.showSess); if (S.overlay) S.overlay.refresh(); }
        else if (id === "ct-ctx") { S.showCtx = !S.showCtx; e.target.classList.toggle("on", S.showCtx); if (S.overlay) S.overlay.refresh(); }
        else if (id === "ct-hline") setTool("h");
        else if (id === "ct-tline") setTool("t");
        else if (id === "ct-clear") { if (S.drawings.length && window.confirm("למחוק את כל " + S.drawings.length + " הקווים שציירת על " + symbol() + "?")) { S.drawings = []; S.selDraw = -1; saveDrawings(); if (S.overlay) S.overlay.refresh(); positionDeleteButton(); } }
        else if (id === "ct-layout") { var ws = document.querySelector(".workspace"); ws.classList.toggle("stacked"); try { localStorage.setItem("sl_layout", ws.classList.contains("stacked") ? "stacked" : "side"); } catch (er) {} }
        else {
            var row = e.target.closest && e.target.closest("tr.trade-row[data-entry]");
            if (row) { jumpTo(parseInt(row.getAttribute("data-entry"), 10)); }
        }
    });
    document.addEventListener("keydown", function (e) {
        var tg = e.target; if (tg && (tg.tagName === "INPUT" || tg.tagName === "TEXTAREA" || tg.isContentEditable)) return;
        var t = tv(); if (!t) return;
        var ts = t.chart.timeScale();
        if (e.key === "ArrowRight") { if (e.shiftKey) ts.scrollToPosition(ts.scrollPosition() + 1, false); else step(1); }
        else if (e.key === "ArrowLeft") { if (e.shiftKey) ts.scrollToPosition(ts.scrollPosition() - 1, false); else step(-1); }
        else if (e.key === "+" || e.key === "=") { ts.applyOptions({barSpacing: Math.min(60, ts.options().barSpacing * 1.25)}); }
        else if (e.key === "-" || e.key === "_") { ts.applyOptions({barSpacing: Math.max(0.5, ts.options().barSpacing / 1.25)}); }
        else if (e.key === "r" || e.key === "R") ts.fitContent();
        else if (e.key === "n" || e.key === "N") ts.scrollToRealTime();
        else if (e.key === "e" || e.key === "E") { var b1 = $("ct-eq"); if (b1) b1.click(); }
        else if (e.key === "s" || e.key === "S") { var b2 = $("ct-sess"); if (b2) b2.click(); }
        else if (e.key === "h" || e.key === "H") setTool("h");
        else if (e.key === "t" || e.key === "T") setTool("t");
        else if (e.key === "Delete" || e.key === "Backspace") { if (S.selDraw >= 0) { e.preventDefault(); deleteSelectedDrawing(); } }
        else if (e.key === "Escape") { setTool(null); S.selEntry = null; S.selIdx = -1; if (S.selDraw >= 0) selectDrawing(-1); updateCounter(); if (S.overlay) S.overlay.refresh(); }
    });
    window.__lab = {state: S, jumpTo: jumpTo, step: step, refresh: function () { fetchIndexAndContext(); fetchTrades(); }};
})();
