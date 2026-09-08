/* תפריט "Indicators" (סרגל עליון, ליד כפתורי הטיים-פריים) - לגמרי בצד לקוח,
   בלי שום סבב לשרת: קורא ישירות מ-window.__tv.candleSeries/volumeSeries.data()
   (מה שכבר מצויר על הגרף הראשי, ראו chart.js), מחשב את האינדיקטור ומצייר
   אותו כ-series נוסף על אותו chart. לכל אינדיקטור מותר עותק אחד בלבד -
   הכפתור "Add" הופך ל-"✕" (הסרה) כל עוד הוא פעיל, בדיוק כמו שהתבקש.

   מזהה שינויי-דאטה ב-polling (לא hook לתוך chart.js) בכוונה: chart.js הוא
   קובץ גדול ומורכב עם הרבה לוגיקת LOD/לייב עדינה - polling על
   candleSeries.data() (שכבר משקף את כל עדכון, מכל מקור: LOD/לייב/החלפת
   סימבול) נותן את אותה תוצאה בלי לגעת שם בכלל.
*/
(function () {
    var POLL_MS = 400;

    // עיצוב פאנל ה-RSI - תואם למראה המדויק שהתבקש (בהתבסס על צילום מסך אמיתי
    // של TradingView): קו RSI סגול, קו סיגנל (SMA של ה-RSI עצמו) זהב, ורצועה
    // עליונה של 18% מגובה הגרף שמוקדשת לפאנל (אותה טכניקה בדיוק כמו הוולום -
    // priceScaleId ייעודי עם scaleMargins, לא multi-pane אמיתי שלא קיים ב-LWC 4.x).
    var RSI_COLOR = "#7B68EE";
    var RSI_SIGNAL_COLOR = "#FFC107";
    var RSI_SIGNAL_LENGTH = 14;
    var RSI_TOP_MARGIN = 0.82;
    var RSI_OB = 70, RSI_MID = 50, RSI_OS = 30;
    // ברירת-המחדל של רצועת הוולום (dashboard/chart.py: VOLUME_SCALE_MARGINS) -
    // חופפת בדיוק לרצועה שה-RSI תופס (שתיהן "תחתית הגרף"). כל עוד RSI פעיל
    // דוחקים את הוולום לרצועה צרה מעליו (RSI_VOLUME_MARGINS_WITH_RSI), ומחזירים
    // להגדרת ברירת-המחדל בהסרה - כדי ש"נקי ונפרד מהוולום" (דרישה מפורשת) יתקיים
    // רק כשה-RSI באמת מוצג, ולא יגזול מקום מהגרף כל הזמן.
    var VOLUME_MARGINS_DEFAULT = {top: 0.78, bottom: 0.0};
    var VOLUME_MARGINS_WITH_RSI = {top: 0.62, bottom: 0.2};

    // הגדרת כל אינדיקטור: איך לבנות/להסיר את ה-series שלו, ואיך לחשב אותו
    // מתוך מערך הנרות (candles - בדיוק הפורמט של candleSeries.data():
    // {time, open, high, low, close}) והוולום (volumes: {time, value}).
    var DEFS = {
        sma: {
            multi: false,
            create: function (chart, p) {
                return addLine(chart, {
                    color: "#2962FF", lineWidth: 2, priceLineVisible: false,
                    lastValueVisible: true, crosshairMarkerVisible: false,
                    title: "SMA " + p.length,
                });
            },
            compute: function (candles, _volumes, p) {
                return smaPoints(candles.map(closeOf), candles, p.length);
            },
        },
        ema: {
            multi: false,
            create: function (chart, p) {
                return addLine(chart, {
                    color: "#AB47BC", lineWidth: 2, priceLineVisible: false,
                    lastValueVisible: true, crosshairMarkerVisible: false,
                    title: "EMA " + p.length,
                });
            },
            compute: function (candles, _volumes, p) {
                return emaPoints(candles.map(closeOf), candles, p.length);
            },
        },
        vwap: {
            multi: false,
            create: function (chart) {
                return addLine(chart, {
                    color: "#FFB300", lineWidth: 2, priceLineVisible: false,
                    lastValueVisible: true, crosshairMarkerVisible: false,
                    title: "VWAP",
                });
            },
            compute: function (candles, volumes) {
                return vwapPoints(candles, volumes);
            },
        },
        rsi: {
            multi: true,
            create: function (chart, p) {
                var scaleId = "indicator-rsi";
                // priceScaleId ייעודי, דחוף לרצועה עליונה של הפאנל (scaleMargins) -
                // אותה טכניקה בדיוק כמו הוולום (ראו chart.js: addHistogram +
                // priceScaleId:"volume"), כדי לדמות "sub-panel" בלי תמיכת
                // multi-pane אמיתית ב-LWC 4.x. lastValueVisible:true על שני
                // הקווים - כך LWC מצייר לבד את התגית הצבעונית עם הערך האחרון
                // על ציר ה-Y הימני (סגול/זהב, בדיוק כמו בצילום המסך), בלי
                // שנצטרך לצייר את זה ידנית.
                var main = addLine(chart, {
                    color: RSI_COLOR, lineWidth: 2, priceScaleId: scaleId,
                    priceLineVisible: false, lastValueVisible: true,
                    crosshairMarkerVisible: true, title: "",
                });
                var signal = addLine(chart, {
                    color: RSI_SIGNAL_COLOR, lineWidth: 1.5, priceScaleId: scaleId,
                    priceLineVisible: false, lastValueVisible: true,
                    crosshairMarkerVisible: false, title: "",
                });
                chart.priceScale(scaleId).applyOptions({
                    scaleMargins: {top: RSI_TOP_MARGIN, bottom: 0},
                    borderVisible: true,
                    borderColor: "#2A2E37",
                });
                // דוחקים את הוולום לרצועה צרה מעל ה-RSI - בלי זה שניהם חופפים
                // לאותה רצועה תחתונה (VOLUME_MARGINS_DEFAULT) ומצטיירים זה על
                // זה. מוחזר ל-remove() למטה.
                if (typeof chart.priceScale === "function") {
                    try { chart.priceScale("volume").applyOptions({scaleMargins: VOLUME_MARGINS_WITH_RSI}); }
                    catch (e) {}
                }
                // קווי-רף אופקיים (70/50/30), מקווקוים ועדינים - ה-API המובנה
                // (createPriceLine) מספיק כאן: הם קבועים ולא תלויי-דאטה, אין
                // סיבה לצייר אותם ידנית על ה-canvas בכל פריים כמו את המילויים.
                var lineStyle = (window.LightweightCharts && LightweightCharts.LineStyle
                    && LightweightCharts.LineStyle.Dashed) || 2;
                var priceLines = [
                    {price: RSI_OB, color: "rgba(150,153,164,0.55)"},
                    {price: RSI_MID, color: "rgba(150,153,164,0.28)"},
                    {price: RSI_OS, color: "rgba(150,153,164,0.55)"},
                ].map(function (l) {
                    return main.createPriceLine({
                        price: l.price, color: l.color, lineWidth: 1,
                        lineStyle: lineStyle, axisLabelVisible: true, title: "",
                    });
                });
                // רקע עדין (גוון סגלגל) + מילוי ירוק/אדום מעל 70 / מתחת ל-30 -
                // primitive מצויר על ה-canvas (בדיוק כמו createTradeMarkers
                // ב-chart.js), כי ל-LWC 4.x אין "מילוי בין שני ערכי-סף" מובנה.
                var fillPrimitive = createRsiFillPrimitive(RSI_TOP_MARGIN, RSI_OB, RSI_OS);
                if (typeof main.attachPrimitive === "function") main.attachPrimitive(fillPrimitive);

                var label = document.createElement("div");
                label.className = "rsi-panel-label";
                var container = document.getElementById("chart-container");
                if (container) container.appendChild(label);

                return {main: main, signal: signal, priceLines: priceLines,
                        fillPrimitive: fillPrimitive, label: label};
            },
            remove: function (chart, series) {
                (series.priceLines || []).forEach(function (pl) {
                    try { series.main.removePriceLine(pl); } catch (e) {}
                });
                chart.removeSeries(series.main);
                chart.removeSeries(series.signal);
                if (series.label && series.label.parentNode) series.label.parentNode.removeChild(series.label);
                // מאפסים את המרווח - בלי זה נשאר "חור" ריק בתחתית הגרף הראשי
                // גם אחרי הסרת ה-RSI, כי ל-LWC 4.x אין דרך למחוק price scale.
                chart.priceScale("indicator-rsi").applyOptions({scaleMargins: {top: 0, bottom: 0}});
                // מחזירים לוולום את הרצועה המלאה שהיה לו לפני שה-RSI דחק אותו.
                try { chart.priceScale("volume").applyOptions({scaleMargins: VOLUME_MARGINS_DEFAULT}); }
                catch (e) {}
            },
            setData: function (series, data) {
                series.main.setData(data.main);
                series.signal.setData(data.signal);
                if (series.fillPrimitive) series.fillPrimitive.setPoints(data.main);
                updateRsiLabel(series.label, data);
            },
            compute: function (candles, _volumes, p) {
                var main = rsiPoints(candles.map(closeOf), candles, p.length);
                var signal = smaOfSeries(main, RSI_SIGNAL_LENGTH);
                return {main: main, signal: signal, length: p.length};
            },
        },
        bb: {
            multi: true,
            create: function (chart, p) {
                var mid = addLine(chart, {
                    color: "#2962FF", lineWidth: 1, priceLineVisible: false,
                    lastValueVisible: false, crosshairMarkerVisible: false,
                    title: "BB " + p.length,
                });
                var opts = {
                    color: "rgba(41,98,255,0.55)", lineWidth: 1, priceLineVisible: false,
                    lastValueVisible: false, crosshairMarkerVisible: false,
                };
                var upper = addLine(chart, opts);
                var lower = addLine(chart, opts);
                return {upper: upper, middle: mid, lower: lower};
            },
            remove: function (chart, series) {
                chart.removeSeries(series.upper);
                chart.removeSeries(series.middle);
                chart.removeSeries(series.lower);
            },
            setData: function (series, data) {
                series.upper.setData(data.upper);
                series.middle.setData(data.middle);
                series.lower.setData(data.lower);
            },
            compute: function (candles, _volumes, p) {
                return bollingerPoints(candles.map(closeOf), candles, p.length, p.std);
            },
        },
    };

    var ACTIVE = {};  // key -> {series, params}

    function closeOf(c) { return c.close; }

    // תואם ל-4.x וגם ל-5.x של Lightweight Charts, אותה מוסכמה בדיוק כמו
    // ה-addLine הפרטי בתוך chart.js (לא נגיש מכאן - כפילות קטנה, לא שינוי
    // בקובץ הקיים).
    function addLine(chart, opts) {
        return typeof chart.addLineSeries === "function"
            ? chart.addLineSeries(opts)
            : chart.addSeries(LightweightCharts.LineSeries, opts);
    }

    // ---------- מתמטיקת האינדיקטורים ----------
    // כולן מחזירות מערך {time, value} מיושר לפי candles - נקודות שאין להן
    // עדיין ערך (למשל SMA לפני שהצטברו length ברים) פשוט מדולגות, לא 0/NaN -
    // כך LWC פשוט לא מצייר שם, בדיוק כמו שהיה עם ה-MA20 הקבועה הקיימת.
    function smaPoints(closes, candles, length) {
        length = Math.max(1, Math.floor(length) || 1);
        var out = [], sum = 0;
        for (var i = 0; i < closes.length; i++) {
            sum += closes[i];
            if (i >= length) sum -= closes[i - length];
            if (i >= length - 1) out.push({time: candles[i].time, value: sum / length});
        }
        return out;
    }

    function emaPoints(closes, candles, length) {
        length = Math.max(1, Math.floor(length) || 1);
        var out = [], k = 2 / (length + 1), prev = null;
        for (var i = 0; i < closes.length; i++) {
            if (i < length - 1) continue;
            if (prev === null) {
                // זריעה: ממוצע פשוט על ה-length הראשונים, בדיוק כמו ההגדרה הרגילה.
                var sum = 0;
                for (var j = i - length + 1; j <= i; j++) sum += closes[j];
                prev = sum / length;
            } else {
                prev = closes[i] * k + prev * (1 - k);
            }
            out.push({time: candles[i].time, value: prev});
        }
        return out;
    }

    function rsiPoints(closes, candles, length) {
        length = Math.max(1, Math.floor(length) || 1);
        var out = [];
        if (closes.length < length + 1) return out;
        var gain = 0, loss = 0;
        for (var i = 1; i <= length; i++) {
            var diff = closes[i] - closes[i - 1];
            if (diff > 0) gain += diff; else loss -= diff;
        }
        var avgGain = gain / length, avgLoss = loss / length;
        out.push(rsiPoint(candles[length].time, avgGain, avgLoss));
        for (var k = length + 1; k < closes.length; k++) {
            var d = closes[k] - closes[k - 1];
            var g = d > 0 ? d : 0, l = d < 0 ? -d : 0;
            avgGain = (avgGain * (length - 1) + g) / length;
            avgLoss = (avgLoss * (length - 1) + l) / length;
            out.push(rsiPoint(candles[k].time, avgGain, avgLoss));
        }
        return out;
    }
    function rsiPoint(time, avgGain, avgLoss) {
        var rs = avgLoss === 0 ? Infinity : avgGain / avgLoss;
        var value = avgLoss === 0 ? 100 : 100 - 100 / (1 + rs);
        return {time: time, value: value};
    }

    // "קו הסיגנל" של RSI - ממוצע-נע פשוט על ה-RSI *עצמו* (לא על המחיר), אותה
    // הגדרת ברירת-מחדל בדיוק כמו "RSI-based MA" של TradingView.
    function smaOfSeries(points, length) {
        length = Math.max(1, Math.floor(length) || 1);
        var out = [], sum = 0;
        for (var i = 0; i < points.length; i++) {
            sum += points[i].value;
            if (i >= length) sum -= points[i - length].value;
            if (i >= length - 1) out.push({time: points[i].time, value: sum / length});
        }
        return out;
    }

    // primitive על ה-canvas של קו ה-RSI (main): רקע עדין לכל פאנל ה-RSI (מבדיל
    // אותו חזותית מהגרף הראשי/הוולום), ומילוי ירוק כשה-RSI מעל obLevel / אדום
    // כשמתחתיו ל-osLevel. הטריק למילוי בין קו לסף בלי API ייעודי ב-LWC 4.x:
    // "הצמדת" (clamp) גובה-הפיקסל של כל נקודה לגובה-הסף כשהיא בצד ה"לא רלוונטי"
    // - כך הפוליגון היחיד שמצטייר מתאפס מעצמו (גובה 0) בדיוק היכן שאין חצייה,
    // בלי לחשב נקודות-חיתוך בין קטעים בעצמנו.
    function createRsiFillPrimitive(topMarginFraction, obLevel, osLevel) {
        var points = [], chartRef = null, seriesRef = null, requestUpdate = null;

        function fillZone(ctx, ts, thresholdY, mode, color) {
            var pts = [];
            for (var i = 0; i < points.length; i++) {
                var x = ts.timeToCoordinate(points[i].time);
                if (x === null) continue;
                var y = seriesRef.priceToCoordinate(points[i].value);
                if (y === null) continue;
                var clamped = mode === "above" ? Math.min(y, thresholdY) : Math.max(y, thresholdY);
                pts.push({x: x, y: clamped});
            }
            if (pts.length < 2) return;
            ctx.beginPath();
            ctx.moveTo(pts[0].x, thresholdY);
            for (var j = 0; j < pts.length; j++) ctx.lineTo(pts[j].x, pts[j].y);
            ctx.lineTo(pts[pts.length - 1].x, thresholdY);
            ctx.closePath();
            ctx.fillStyle = color;
            ctx.fill();
        }

        var renderer = {
            draw: function (target) {
                target.useMediaCoordinateSpace(function (scope) {
                    if (!chartRef || !seriesRef) return;
                    var ctx = scope.context;
                    var panelTop = scope.mediaSize.height * topMarginFraction;

                    ctx.save();
                    ctx.fillStyle = "rgba(90,70,160,0.06)";
                    ctx.fillRect(0, panelTop, scope.mediaSize.width, scope.mediaSize.height - panelTop);

                    if (points.length) {
                        var ts = chartRef.timeScale();
                        var yOB = seriesRef.priceToCoordinate(obLevel);
                        var yOS = seriesRef.priceToCoordinate(osLevel);
                        if (yOB !== null) fillZone(ctx, ts, yOB, "above", "rgba(38,90,50,0.6)");
                        if (yOS !== null) fillZone(ctx, ts, yOS, "below", "rgba(120,30,30,0.6)");
                    }
                    ctx.restore();
                });
            },
        };

        var paneView = {
            renderer: function () { return renderer; },
            zOrder: function () { return "bottom"; },  // מתחת לקו ה-RSI/הסיגנל/הקווים המקווקוים
        };

        return {
            attached: function (param) {
                chartRef = param.chart; seriesRef = param.series; requestUpdate = param.requestUpdate;
            },
            detached: function () { chartRef = null; seriesRef = null; requestUpdate = null; },
            paneViews: function () { return [paneView]; },
            updateAllViews: function () {},
            setPoints: function (pts) { points = pts || []; if (requestUpdate) requestUpdate(); },
        };
    }

    // מעדכן את תווית "RSI 14 close" בפינה השמאלית-עליונה של הפאנל - טקסט
    // ה-DOM (לא canvas) כדי לערבב בקלות שני צבעים בתוך שורה אחת.
    function updateRsiLabel(label, data) {
        if (!label) return;
        var container = document.getElementById("chart-container");
        if (container) label.style.top = (container.clientHeight * RSI_TOP_MARGIN + 6) + "px";
        var lastMain = data.main.length ? data.main[data.main.length - 1].value : null;
        var lastSignal = data.signal.length ? data.signal[data.signal.length - 1].value : null;
        label.innerHTML =
            '<span class="rsi-label-title">RSI ' + data.length + ' close</span>' +
            '<span class="rsi-label-main">' + (lastMain === null ? "—" : lastMain.toFixed(2)) + '</span>' +
            '<span class="rsi-label-signal">' + (lastSignal === null ? "—" : lastSignal.toFixed(2)) + '</span>';
    }

    function bollingerPoints(closes, candles, length, stdMult) {
        length = Math.max(1, Math.floor(length) || 1);
        stdMult = stdMult || 2;
        var upper = [], middle = [], lower = [];
        for (var i = length - 1; i < closes.length; i++) {
            var sum = 0;
            for (var j = i - length + 1; j <= i; j++) sum += closes[j];
            var mean = sum / length;
            var variance = 0;
            for (var k = i - length + 1; k <= i; k++) variance += Math.pow(closes[k] - mean, 2);
            var sd = Math.sqrt(variance / length);
            var t = candles[i].time;
            middle.push({time: t, value: mean});
            upper.push({time: t, value: mean + stdMult * sd});
            lower.push({time: t, value: mean - stdMult * sd});
        }
        return {upper: upper, middle: middle, lower: lower};
    }

    // VWAP מתאפס כל יום מסחר (session) - אותה הגדרה בדיוק כמו self.vwap
    // בבקטסט (dashboard/strategy.py: session_vwap). "יום" מזוהה לפי חצות
    // בשעון התצוגה (candle.time הוא כבר ה"UTC מזויף" של שעון ישראל, ראו
    // data.py: _epoch) - חלוקה שלמה ב-86400 שניות נותנת את היום הקלנדרי הנכון.
    function vwapPoints(candles, volumes) {
        var volByTime = {};
        for (var i = 0; i < volumes.length; i++) volByTime[volumes[i].time] = volumes[i].value || 0;
        var out = [], cumPV = 0, cumV = 0, curDay = null;
        for (var k = 0; k < candles.length; k++) {
            var c = candles[k];
            var day = Math.floor(c.time / 86400);
            if (day !== curDay) { curDay = day; cumPV = 0; cumV = 0; }
            var typical = (c.high + c.low + c.close) / 3;
            var vol = volByTime[c.time] || 0;
            cumPV += typical * vol;
            cumV += vol;
            if (cumV > 0) out.push({time: c.time, value: cumPV / cumV});
        }
        return out;
    }

    // ---------- ניהול מצב: הוספה/הסרה/רענון ----------
    function getCandles() {
        return (window.__tv && window.__tv.candleSeries) ? window.__tv.candleSeries.data() : [];
    }
    function getVolumes() {
        return (window.__tv && window.__tv.volumeSeries) ? window.__tv.volumeSeries.data() : [];
    }

    function paramsFor(key) {
        // querySelectorAll('.indicator-param-input') לא עובד - Dash עוטף
        // dcc.Input(type="number") ב-div (עם כפתורי +/- stepper) שנושא את
        // ה-className שנתנו, לא ה-<input> עצמו. שאילתה ישירה לפי תחילית ה-id
        // (ייחודי לכל אינדיקטור+פרמטר) עוקפת את זה, בלי תלות במבנה-העטיפה.
        var params = {};
        document.querySelectorAll('input[id^="ind-' + key + '-"]').forEach(function (inp) {
            var pname = inp.id.slice(("ind-" + key + "-").length);
            params[pname] = parseFloat(inp.value);
        });
        return params;
    }

    function computeAndSet(key) {
        var def = ACTIVE[key];
        if (!def || !window.__tv || !window.__tv.chart) return;
        var candles = getCandles();
        if (!candles.length) return;
        var data = DEFS[key].compute(candles, getVolumes(), def.params);
        if (DEFS[key].setData) DEFS[key].setData(def.series, data);
        else def.series.setData(data);
    }

    function refreshAll() {
        Object.keys(ACTIVE).forEach(computeAndSet);
    }

    function setRowState(key, active) {
        var addBtn = document.getElementById("ind-" + key + "-add");
        var removeBtn = document.getElementById("ind-" + key + "-remove");
        if (addBtn) addBtn.style.display = active ? "none" : "";
        if (removeBtn) removeBtn.style.display = active ? "" : "none";
    }

    function addIndicator(key) {
        if (ACTIVE[key]) return;  // מותר עותק אחד בלבד
        if (!window.__tv || !window.__tv.chart) return;
        var def = DEFS[key];
        var params = paramsFor(key);
        var series = def.create(window.__tv.chart, params);
        ACTIVE[key] = {series: series, params: params};
        computeAndSet(key);
        setRowState(key, true);
    }

    function removeIndicator(key) {
        var active = ACTIVE[key];
        if (!active) return;
        var def = DEFS[key];
        if (def.remove) def.remove(window.__tv.chart, active.series);
        else window.__tv.chart.removeSeries(active.series);
        delete ACTIVE[key];
        setRowState(key, false);
    }

    // ---------- תפריט: פתיחה/סגירה + כפתורי Add/Remove ----------
    function toggleMenu(force) {
        var menu = document.getElementById("indicators-menu");
        if (!menu) return;
        var show = force !== undefined ? force : menu.style.display === "none";
        menu.style.display = show ? "block" : "none";
    }

    document.addEventListener("click", function (e) {
        if (e.target.closest("#indicators-btn")) { toggleMenu(); return; }
        var addBtn = e.target.closest(".indicator-add-btn");
        if (addBtn) {
            var row = addBtn.closest(".indicator-row");
            if (row) addIndicator(row.getAttribute("data-key"));
            return;
        }
        var removeBtn = e.target.closest(".indicator-remove-btn");
        if (removeBtn) {
            var row2 = removeBtn.closest(".indicator-row");
            if (row2) removeIndicator(row2.getAttribute("data-key"));
            return;
        }
        if (!e.target.closest("#indicators-menu") && !e.target.closest("#indicators-btn")) {
            toggleMenu(false);
        }
    });
    document.addEventListener("keydown", function (e) {
        if (e.key === "Escape") toggleMenu(false);
    });

    // ---------- זיהוי שינוי-דאטה (LOD/לייב/סימבול חדש) ----------
    var lastSig = "";
    setInterval(function () {
        if (!Object.keys(ACTIVE).length) return;
        var candles = getCandles();
        if (!candles.length) return;
        var last = candles[candles.length - 1];
        var sig = candles.length + "|" + last.time + "|" + last.close;
        if (sig !== lastSig) { lastSig = sig; refreshAll(); }
    }, POLL_MS);
})();
