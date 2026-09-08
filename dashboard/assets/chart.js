/* מנוע הציור: מקבל מטען מ-Dash (dcc.Store) ומצייר אותו ב-TradingView Lightweight Charts.
   הגרף נוצר פעם אחת ואחר כך רק מתעדכן — יצירה מחדש בכל שינוי הייתה מאבדת את הזום. */
(function () {
    var chart = null, candleSeries = null, volumeSeries = null, markerHandle = null;
    var tradeMarkers = null;   // ה-primitive שמצייר את משולשי העסקאות על המחיר המדויק
    var smaSeries = null;      // קו ממוצע נע 20 — מחושב כאן, מאותו דאטה שכבר נטען לגרף
    var closeHistory = [];     // {time, close} לכל הנרות שנטענו, כדי לחשב SMA בלי לשאול את הספרייה
    var SMA_PERIOD = 20;
    var SMA_COLOR = "#F5A623";
    var lastKey = null;  // symbol|timeframe — החלפה שלו מצדיקה איפוס הזום
    // כיול התצוגה: pendingFit נשאר דלוק עד שמתבצע כיול אחד מוצלח על מידות אמיתיות,
    // ו-fitDeadline מאפשר כיול נוסף בחלון קצר אחריו כדי לספוג התייצבות של הפריסה.
    var pendingFit = false, fitDeadline = 0;
    // טיימר סגירת הנר: השרת שולח את מספר השניות שנותרו (מחושב בשעון הבורסה), וכאן
    // רק סופרים לאחור בין עדכון לעדכון כדי שהתצוגה תזוז חלק כל שנייה.
    var countdownEl = null, secondsLeft = null, countdownTimer = null;
    var resetLiveBtn = null;

    // מציג/מחביא את כפתור "חזרה לעכשיו" - כמו ב-TradingView, רק כשגללו מספיק
    // הרחק מהקצה החי. scrollPosition() של LWC מחזיר מרחק בברים מהקצה הימני *של
    // הדאטה עצמה* - בקצה החי, בלי שום גלילה, זה כבר שווה ל-rightOffset (6 כאן,
    // ראו chart.py) ולא 0 - סף קבוע נמוך (למשל 2) הראה את הכפתור תמיד, גם
    // בטעינה ראשונית. משווים מול rightOffset עצמו בתוספת מרווח קטן.
    function updateResetLiveVisibility() {
        if (!resetLiveBtn || !chart) return;
        var pos;
        try { pos = chart.timeScale().scrollPosition(); } catch (e) { return; }
        var rightOffset = chart.timeScale().options().rightOffset || 0;
        // scrollPosition() סוטה מ-rightOffset לשני הכיוונים: שלילי כשגוללים שמאלה
        // אל תוך ההיסטוריה (למשל 2020), חיובי כשגוללים ימינה מעבר לקצה החי (עתיד).
        // Math.abs תופס את שני המקרים בבדיקה אחת - הקודם (pos > rightOffset + 3)
        // תפס רק סטייה חיובית, ולכן הכפתור הופיע רק בגלילה ימינה, לא שמאלה.
        resetLiveBtn.style.display = (typeof pos === "number" && Math.abs(pos - rightOffset) > 3) ? "flex" : "none";
    }
    var lastBarTime = null;  // הזמן של הנר האחרון שכבר מצויר — series.update דוחה זמן ישן ממנו
    var lastClose = null;    // המחיר האחרון — למיקום ה-badge של הספירה לאחור על ציר ה-Y
    var lastOpen = null;     // הפתיחה של הנר הנוכחי — קובעת אם הספירה לאחור צבועה ירוק/אדום
    var currentTimeframe = null;
    var currentSymbol = null;

    // מדרג-פירוט (LOD, כמו TradingView): allCandles/allVolume הם עותק מקומי של
    // מה שמצויר *כרגע* - לא כל ההיסטוריה, רק חלון ברזולוציה (viewTier) שמתאימה
    // לרוחב הטווח הנראה. viewWindow הם הגבולות שהחלון הזה כבר מכסה; כשגוללים/
    // מזזמים מחוץ להם, או שרוחב הטווח כבר מצדיק רזולוציה גסה/עדינה אחרת, מבקשים
    // חלון חדש (requestLodWindow) שמחליף - לא מצטבר על - את מה שהיה. כך שום
    // פעולה לא דוחפת מיליוני נרות לדפדפן, בלי קשר לכמה היסטוריה יש בקאש.
    var allCandles = [], allVolume = [];
    var viewTier = null, viewWindow = {from: null, to: null};
    var lodRequestSeq = 0;   // מתעלמים מתשובות LOD מיושנות (בקשה חדשה יותר כבר יצאה)
    // מרכז הבקשה האחרונה שבאמת נשלחה - ראו onVisibleTimeRangeChanged, בלימת שיטפון
    // בקשות ליד קצה אמיתי (עכשיו/סוף הדאטה ההיסטורי).
    var lastRequestedCenter = null;
    // סימוני-עסקאות (משולשים): true רק אחרי "Add to chart" מוצלח, על אותו
    // סימבול/טיים-פריים - ראו render (payload.backtestReady) ו-requestMarkersWindow.
    var markersAvailable = false;
    var markersRequestSeq = 0;
    var markersDebounceTimer = null;
    // resolve לכל seq בנפרד, לא משתנה גלובלי יחיד - אחרת בקשה חדשה (למשל
    // מהגלילה התגובתית) שמגיעה בזמן ש"Go to date" עוד מחכה לתשובה שלו, הייתה
    // דורסת את ה-resolve שלו ותוקעת אותו לצמיתות (בדיוק ה"טוען..." שלא נגמר).
    var lodResolvers = {};
    var lodDebounceTimer = null;

    // גבולות הרזולוציה: אף פעם לא עדין יותר מהטיים-פריים שנבחר בפועל (לחיצה על
    // "1h" לא אמורה להראות פתאום נרות דקה גם בזום-אין), ולא נבחר tier שבו יש
    // יותר מ-MAX_BARS_PER_VIEW נרות בטווח הנראה. הורד מ-200,000 ל-100,000 כשה-
    // buffer בשרת גדל (load_lod_window: 3x הטווח במקום 1.5x, כדי שגלילה רציפה
    // לא תבקש חלון חדש על כל תזוזה קטנה) - כך שהחלון המלא שנשלף בפועל (טווח
    // נראה + buffer משני הצדדים, עד פי 7) נשאר קרוב למה שכבר נבדק כמהיר בלי
    // הקפאה, ולא קופץ למיליון+ נרות בבקשה אחת.
    var TIER_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800,
                         "1h": 3600, "4h": 14400, "1d": 86400, "1w": 604800};
    var TIER_ORDER = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"];
    var MAX_BARS_PER_VIEW = 100000;

    // תואם ל-UP/DOWN ב-dashboard/chart.py (CANDLE_OPTIONS), כדי שצבע ה-badge יהיה
    // בדיוק צבע הנר עצמו ולא רק "בערך אדום/ירוק". ירוק/אדום "זוהרים" על רקע שחור.
    var UP_COLOR = "#00E676", DOWN_COLOR = "#FF1744";

    // dashboard/data.py ו-chart.py שולחים "u"/"d" בלבד במקום מחרוזת rgba מלאה על
    // כל בר וולום - חוסך מגה-בייטים על חלון היסטוריה גדול. מרחיבים חזרה כאן,
    // ממש לפני שמזינים ל-volumeSeries (ש-LightweightCharts דורש ממנה צבע אמיתי).
    // אותם RGB כמו הנרות (0,230,118 / 255,23,68), אטימות מוגברת שלא ייראו שטוחות.
    var VOL_COLOR = {u: "rgba(0,230,118,0.65)", d: "rgba(255,23,68,0.65)"};
    function decodeVolume(v) {
        return {time: v.time, value: v.value, color: VOL_COLOR[v.color] || v.color};
    }

    // תואם ל-_BAR_SECONDS ב-ib_data.py. משמש לחישוב שעת הסגירה של הנר הנוכחי.
    var BAR_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800,
                        "1h": 3600, "4h": 14400, "1d": 86400, "1w": 604800};

    // הערה: הספירה-לאחור בפועל מיושמת למטה (positionCountdown/startCountdown),
    // מונעת ע"י secondsToClose שהשרת מחשב (dashboard/live.py: seconds_to_close),
    // ומצוירת לתוך countdownEl שנוצר דינמית ב-ensureChart. בלוק ישן שהיה כאן קודם
    // עדכן div נפרד (#bar-countdown הסטטי מ-app.py) לפי שעון ניו-יורק בלבד, בלי
    // ההמרה לשעון ישראל — שני הריבועים היו חופפים על המסך וזה נראה "לא מדויק".
    // הוסר; אם צריך גם למחוק את ה-div הריק מ-app.py זה בטוח (אין יותר קוד שנוגע בו).

    // Lightweight Charts שינו את ה-API בין v4 ל-v5. שתי העטיפות האלה מאפשרות לקוד
    // לעבוד עם שתי הגרסאות, כדי ששדרוג CDN לא ישבור את הגרף בלי אזהרה.
    function addCandles(c, opts) {
        return typeof c.addCandlestickSeries === "function"
            ? c.addCandlestickSeries(opts)
            : c.addSeries(LightweightCharts.CandlestickSeries, opts);
    }
    function addHistogram(c, opts) {
        return typeof c.addHistogramSeries === "function"
            ? c.addHistogramSeries(opts)
            : c.addSeries(LightweightCharts.HistogramSeries, opts);
    }
    function addLine(c, opts) {
        return typeof c.addLineSeries === "function"
            ? c.addLineSeries(opts)
            : c.addSeries(LightweightCharts.LineSeries, opts);
    }

    // ---------- ממוצע נע 20 (SMA) ----------
    // מחושב כאן מ-closeHistory (עותק מקומי של מחירי הסגירה), לא נשאב בנפרד — אותו
    // דאטה בדיוק שכבר הגיע מ-IB לגרף עצמו. חלון-נגרר (sliding sum) לחישוב המלא
    // כדי שגם 5 שנות נרות דקה (מאות אלפי נקודות) יחושבו בפעם אחת בלי לגרור.
    function computeFullSMA(history, period) {
        var out = [];
        var sum = 0;
        for (var i = 0; i < history.length; i++) {
            sum += history[i].close;
            if (i >= period) sum -= history[i - period].close;
            if (i >= period - 1) out.push({time: history[i].time, value: sum / period});
        }
        return out;
    }

    // רק הנקודה האחרונה — לעדכון לייב זול (O(period), לא O(n) על כל ההיסטוריה).
    function lastSMAPoint(history, period) {
        var n = history.length;
        if (n < period) return null;  // כמו ב-TradingView: אין קו עד שיש מספיק נרות
        var sum = 0;
        for (var i = n - period; i < n; i++) sum += history[i].close;
        return {time: history[n - 1].time, value: sum / period};
    }
    /* סימוני העסקאות מצוירים ידנית על ה-canvas, ולא דרך series.setMarkers.
       הסיבה: סימונים מובנים נצמדים לנר (aboveBar / belowBar / inBar) ואין להם מושג
       של מחיר — כך שאי אפשר לסמן בהם את מחיר הכניסה/היציאה המדויק. primitive של
       הסדרה מקבל גישה להמרות timeToCoordinate/priceToCoordinate, ולכן יכול לצייר
       את חוד המשולש בדיוק על גובה המחיר. */
    function createTradeMarkers() {
        var points = [], chartRef = null, seriesRef = null, requestUpdate = null;
        var TRI_W = 6, TRI_H = 9;

        var renderer = {
            draw: function (target) {
                target.useMediaCoordinateSpace(function (scope) {
                    if (!chartRef || !seriesRef || !points.length) return;
                    var ctx = scope.context;
                    var ts = chartRef.timeScale();
                    // בצפיפות גבוהה הכיתובים נמרחים לכתם לבן; מציגים אותם רק כשיש מקום.
                    var showText = ts.options().barSpacing >= 6;

                    ctx.save();
                    ctx.textAlign = "center";
                    ctx.textBaseline = "top";
                    ctx.font = "600 10px -apple-system, BlinkMacSystemFont, sans-serif";

                    for (var i = 0; i < points.length; i++) {
                        var p = points[i];
                        var x = ts.timeToCoordinate(p.time);
                        if (x === null) continue;               // מחוץ לטווח הנראה
                        var y = seriesRef.priceToCoordinate(p.price);
                        if (y === null) continue;

                        var up = p.dir === "up";
                        ctx.fillStyle = p.color;
                        ctx.beginPath();
                        ctx.moveTo(x, y);                        // החוד — בדיוק על המחיר
                        if (up) {
                            ctx.lineTo(x - TRI_W, y + TRI_H);
                            ctx.lineTo(x + TRI_W, y + TRI_H);
                        } else {
                            ctx.lineTo(x - TRI_W, y - TRI_H);
                            ctx.lineTo(x + TRI_W, y - TRI_H);
                        }
                        ctx.closePath();
                        ctx.fill();

                        if (showText && p.text) {
                            ctx.fillStyle = "#FFFFFF";
                            ctx.fillText(p.text, x, up ? y + TRI_H + 2 : y + 3);
                        }
                    }
                    ctx.restore();
                });
            }
        };

        var paneView = {
            renderer: function () { return renderer; },
            zOrder: function () { return "top"; }
        };

        return {
            attached: function (param) {
                chartRef = param.chart;
                seriesRef = param.series;
                requestUpdate = param.requestUpdate;
            },
            detached: function () { chartRef = null; seriesRef = null; requestUpdate = null; },
            paneViews: function () { return [paneView]; },
            updateAllViews: function () {},
            setPoints: function (pts) {
                points = pts || [];
                if (requestUpdate) requestUpdate();
            }
        };
    }

    function applyMarkers(series, markers) {
        if (tradeMarkers) { tradeMarkers.setPoints(markers); return; }

        // גיבוי לגרסאות ללא primitives: סימונים מובנים, נצמדים לנר ולא למחיר.
        var fallback = (markers || []).map(function (p) {
            return {time: p.time, position: "belowBar", color: p.color,
                    shape: p.dir === "up" ? "arrowUp" : "arrowDown", text: p.text};
        });
        if (typeof series.setMarkers === "function") { series.setMarkers(fallback); return; }
        if (LightweightCharts.createSeriesMarkers) {
            markerHandle = LightweightCharts.createSeriesMarkers(series, fallback);
        }
    }

    // כיול על מיכל חסר-מידות מייצר טווח מנוון (barSpacing נתקע ברצפה והנרות
    // נדחסים לפינה), ומכיוון שהוא "מוצלח" מבחינת הספרייה הוא נשאר כך. לכן מכיילים
    // רק כשיש מידות אמיתיות, וממשיכים לנסות עד שמתקבלות.
    function usableSize() {
        var el = document.getElementById("chart-container");
        return !!el && el.clientWidth > 50 && el.clientHeight > 50;
    }

    function maybeFit() {
        if (!pendingFit && Date.now() >= fitDeadline) return;
        if (!usableSize()) return;
        chart.timeScale().fitContent();
        pendingFit = false;
    }

    // כותרת ה-OHLC/מספר-הברים/טווח-התאריכים למעלה מחושבת ב-Python רק בטעינה
    // הראשונית (load_chart) - מדרג-הפירוט (LOD) הוא client-side טהור ולא נוגע
    // בה, אז בלי זה היא הייתה נשארת תקועה על הערכים של הטעינה הראשונית לנצח
    // (בדיוק מה שנראה בצילומי המסך: "10,860 bars · Aug 24 → Sep 02" גם אחרי
    // גלילה/קפיצה לתקופה אחרת לגמרי). מחשבים כאן מ-allCandles, באותה מוסכמה
    // בדיוק כמו ohlc_summary/status ב-app.py.
    function fmtHeaderDate(epochSec) {
        return new Date(epochSec * 1000).toLocaleDateString("en-US", {
            month: "short", day: "2-digit", year: "numeric", timeZone: "UTC",
        });
    }
    function updateHeaderText() {
        if (!allCandles.length) return;
        var last = allCandles[allCandles.length - 1];
        var first = allCandles[0];
        var prevClose = allCandles.length > 1 ? allCandles[allCandles.length - 2].close : last.open;
        var chg = last.close - prevClose;
        var pct = prevClose ? (chg / prevClose * 100) : 0;
        var sign = function (n) { return n >= 0 ? "+" : ""; };

        var ohlcLine = document.getElementById("ohlc-line");
        if (ohlcLine && currentSymbol && currentTimeframe) {
            ohlcLine.innerHTML = "<b>" + currentSymbol + " · " + currentTimeframe + "</b>   " +
                "O" + last.open.toFixed(2) + "  H" + last.high.toFixed(2) + "  " +
                "L" + last.low.toFixed(2) + "  C" + last.close.toFixed(2) + "  " +
                sign(chg) + chg.toFixed(2) + " (" + sign(pct) + pct.toFixed(2) + "%)";
        }
        var statusEl = document.getElementById("status");
        if (statusEl) {
            var years = (last.time - first.time) / (365.25 * 86400);
            statusEl.textContent = allCandles.length.toLocaleString("en-US") + " bars · " +
                fmtHeaderDate(first.time) + " → " + fmtHeaderDate(last.time) + " (" + years.toFixed(1) + "y)";
        }
    }

    // עמעום קל של הגרף כל עוד יש בקשת LOD אחת או יותר "באוויר" - כדי שגלילה
    // שמחכה לתשובה תרגיש כמו "בתהליך", לא כמו שהיא פשוט לא הגיבה. מונה ולא
    // בוליאני כדי לטפל נכון בכמה בקשות חופפות (למשל גלילה תגובתית + "Go to").
    var lodLoadingCount = 0;
    function setLodLoading(loading) {
        lodLoadingCount = Math.max(0, lodLoadingCount + (loading ? 1 : -1));
        var el = document.getElementById("chart-container");
        if (!el) return;
        el.style.transition = "opacity 0.15s ease";
        el.style.opacity = lodLoadingCount > 0 ? "0.55" : "1";
    }

    function fmtCountdown(sec) {
        sec = Math.max(0, Math.floor(sec));
        var h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
        var pad = function (n) { return n < 10 ? "0" + n : "" + n; };
        return h > 0 ? h + ":" + pad(m) + ":" + pad(s) : pad(m) + ":" + pad(s);
    }

    // הטיימר יושב על סרגל המחירים, ממש מתחת לתווית המחיר האחרון — כמו ב-TradingView.
    // המיקום האנכי נגזר מהמחיר האחרון בכל טיק, כדי שיישאר צמוד לתווית גם כשהמחיר זז.
    function positionCountdown() {
        if (!countdownEl || !chart || !candleSeries) return;
        var el = document.getElementById("chart-container");
        if (!el || lastClose === null || secondsLeft === null) {
            if (countdownEl) countdownEl.style.display = "none";
            return;
        }
        var y = candleSeries.priceToCoordinate(lastClose);
        if (y === null) { countdownEl.style.display = "none"; return; }
        var scaleWidth = chart.priceScale("right").width();
        countdownEl.style.display = "block";
        countdownEl.style.width = scaleWidth + "px";
        countdownEl.style.top = Math.round(y + 11) + "px";
        countdownEl.textContent = fmtCountdown(secondsLeft);
        // צבע הריבוע לפי כיוון הנר הנוכחי — ירוק אם עולה (close >= open), אדום אם יורד.
        if (lastOpen !== null) {
            countdownEl.style.background = lastClose >= lastOpen ? UP_COLOR : DOWN_COLOR;
            countdownEl.style.color = "#FFFFFF";
        }
    }

    function startCountdown() {
        if (countdownTimer) return;
        countdownTimer = setInterval(function () {
            if (secondsLeft === null) return;
            secondsLeft = Math.max(0, secondsLeft - 1);
            positionCountdown();
        }, 1000);
    }

    function ensureChart(el, payload) {
        if (chart) return;
        chart = LightweightCharts.createChart(el, Object.assign(
            {width: el.clientWidth, height: el.clientHeight}, payload.options));
        resetLiveBtn = document.getElementById("reset-live-btn");
        candleSeries = addCandles(chart, payload.candleOptions);
        volumeSeries = addHistogram(chart, payload.volumeOptions);
        chart.priceScale("volume").applyOptions({scaleMargins: payload.volumeScaleMargins});
        smaSeries = addLine(chart, {
            color: SMA_COLOR, lineWidth: 2, priceLineVisible: false,
            lastValueVisible: true, crosshairMarkerVisible: false, title: "MA 20",
        });

        // הגרף לא מגיב לשינוי גודל מעצמו — בלי זה הוא נחתך כשמשנים חלון.
        // הרינדור הראשון קורה לפני שה-flex מייצב את מידות האזור, ולכן מכיילים שוב
        // בחלון קצר אחרי טעינת דאטה חדשה — ורק בו. קודם היה כאן דגל שנשאר דלוק עד
        // שהמשתמש נוגע בגרף, וכך כל שינוי רוחב של ציר המחיר (שקורה בכל טיק לייב)
        // קרא ל-fitContent וביטל את הזום — הגרף היה "קופא" על כל הטווח.
        new ResizeObserver(function () {
            if (!chart || !usableSize()) return;
            chart.applyOptions({width: el.clientWidth, height: el.clientHeight});
            maybeFit();
            positionCountdown();
        }).observe(el);

        if (typeof candleSeries.attachPrimitive === "function") {
            tradeMarkers = createTradeMarkers();
            candleSeries.attachPrimitive(tradeMarkers);
        }

        // מדרג-פירוט: כל שינוי בטווח הנראה (זום או גלילה) עשוי לחייב רזולוציה
        // אחרת או חלון אחר - onVisibleTimeRangeChanged (למטה) מחליט אם צריך
        // לבקש דאטה חדש. throttle+debounce ולא debounce טהור: זום/גלילה מהירים
        // ורציפים (למשל גלגלת עכבר שממשיכה לזוז) היו מאפסים debounce טהור על כל
        // אירוע ואף פעם לא יורים בקשה עד שהתנועה נעצרת לגמרי - וכל הזמן הזה
        // הגרף כבר קפץ (LWC מצייר את הטווח החדש מיד, לפני שה-JS כאן בכלל רץ)
        // אבל בלי דאטה שמכסה אותו, כך שהמסך נשאר ריק לכל אורך הגלילה. עם
        // throttle, בקשה יוצאת לפחות כל LOD_THROTTLE_MS גם באמצע תנועה רציפה,
        // כך שדאטה ממשיך לזרום פנימה בהדרגה במקום לחכות לעצירה מלאה.
        var lodLastFetchAt = 0;
        var LOD_THROTTLE_MS = 100;
        chart.timeScale().subscribeVisibleTimeRangeChange(function (range) {
            if (lodDebounceTimer) clearTimeout(lodDebounceTimer);
            var now = Date.now();
            if (now - lodLastFetchAt >= LOD_THROTTLE_MS) {
                lodLastFetchAt = now;
                onVisibleTimeRangeChanged(range);
            }
            // תמיד גם מתזמנים קריאה אחרונה קצת אחרי - לתפוס את המצב הסופי המדויק
            // ברגע שהתנועה נרגעת, גם אם ה-throttle כבר ירה לאחרונה ממש לפני זה.
            lodDebounceTimer = setTimeout(function () {
                lodDebounceTimer = null;
                lodLastFetchAt = Date.now();
                onVisibleTimeRangeChanged(range);
            }, 150);
            updateResetLiveVisibility();
        });

        countdownEl = document.createElement("div");
        countdownEl.className = "bar-countdown";
        countdownEl.style.display = "none";
        el.appendChild(countdownEl);
        startCountdown();

        // ידית לניפוי שגיאות ולבדיקות מהקונסולה (window.__tv.chart / .candleSeries)
        window.__tv = {
            chart: chart, candleSeries: candleSeries, volumeSeries: volumeSeries,
            getViewTier: function () { return viewTier; },
            getViewWindow: function () { return viewWindow; },
            triggerLod: function (from, to) { onVisibleTimeRangeChanged({from: from, to: to}); },
            debugState: function () {
                return {currentTimeframe: currentTimeframe, currentSymbol: currentSymbol,
                        maxBars: MAX_BARS_PER_VIEW, lodSeq: lodRequestSeq,
                        hasPrimitiveMarkers: !!tradeMarkers, markersAvailable: markersAvailable};
            },
            requestMarkersWindow: requestMarkersWindow,
            idealTierFor: idealTierFor,
        };
    }

    // Enter בשדה הסימבול: הרכיב של Dash מבצע commit ביציאה מהשדה בלבד, ולכן Enter
    // לבדו לא היה טוען כלום. הופכים אותו ל-blur כדי שההתנהגות תהיה הצפויה.
    document.addEventListener("keydown", function (e) {
        if (e.key === "Enter" && e.target && e.target.id === "symbol-input") {
            e.target.blur();
        }
    });

    // לחיצה על לשונית שכבר פעילה (Strategy Tester / Pine Editor) מקפלת/פותחת את
    // הפאנל, כמו אקורדיון — בנוסף לכפתור המזעור הייעודי. Dash לא שולח callback
    // כש-value של dcc.Tabs לא משתנה (לחיצה על הלשונית הנבחרת כבר), אז זה חייב
    // להיתפס כאן ולהפעיל את כפתור המזעור הקיים, כדי לא לשכפל את לוגיקת הפתיחה/סגירה.
    // mousedown ולא click: חייבים לבדוק את ה-class *לפני* שה-React של Dash מעדכן
    // אותו בעצמו על אותה לחיצה — אחרת מעבר ללשונית חדשה (לא הפעילה) נראה "כבר נבחר"
    // באותו טיק ומקפל את הפאנל בטעות במקום רק להחליף לשונית.
    document.addEventListener("mousedown", function (e) {
        var tab = e.target.closest(".panel-tabs .ptab");
        if (!tab || !tab.classList.contains("ptab-sel")) return;
        var toggleBtn = document.getElementById("panel-toggle");
        if (toggleBtn) toggleBtn.click();
    });

    // ---------- מדרג-פירוט (LOD): בוחר רזולוציה וחלון לפי מה שבאמת נראה ----------

    // הרזולוציה העדינה ביותר שעדיין נותנת פחות מ-MAX_BARS_PER_VIEW נרות על
    // הטווח הנראה, לא עדינה יותר מהטיים-פריים שנבחר בכפתורים.
    function idealTierFor(spanSeconds) {
        var startIdx = TIER_ORDER.indexOf(currentTimeframe);
        if (startIdx < 0) startIdx = 0;
        for (var i = startIdx; i < TIER_ORDER.length; i++) {
            var t = TIER_ORDER[i];
            if (spanSeconds / TIER_SECONDS[t] <= MAX_BARS_PER_VIEW) return t;
        }
        return TIER_ORDER[TIER_ORDER.length - 1];
    }

    // מבקש חלון חדש מיד: fetch() ישיר ל-/api/lod (נתיב Flask, לא Dash callback -
    // ראו app.py) - לא עוד Store+כפתור מוסתר, כדי לחסוך את שכבת ה-serialization/
    // dispatch של Dash על הנתיב הכי חם באפליקציה. תוצאה חוזרת דרך applyLodWindow
    // למטה. seq עולה בכל בקשה כדי שתשובה מיושנת (מבקשה שכבר "נעקפה" על ידי בקשה
    // חדשה יותר) תיפסל אוטומטית. onDone אופציונלי - "Go to date" מחכה לו
    // (requestLodWindowAsync) לפני שהוא מזיז את הטווח הנראה, כי setVisibleRange
    // לטווח שאין לו שום חפיפה עם הדאטה הקיימת לא תמיד מפעיל בעצמו את
    // subscribeVisibleTimeRangeChange בצורה אמינה - עדיף לוודא שהדאטה כבר נטענה
    // ורק אז לזוז אליה, בדיוק כמו טעינה ראשונית רגילה.
    function requestLodWindow(tier, from, to, span, onDone) {
        var mySeq = ++lodRequestSeq;
        if (onDone) lodResolvers[mySeq] = onDone;
        setLodLoading(true);
        var url = "/api/lod?symbol=" + encodeURIComponent(currentSymbol || "") +
            "&tier=" + encodeURIComponent(tier) +
            "&center=" + Math.round((from + to) / 2) +
            "&span=" + Math.round(span) +
            "&seq=" + mySeq;
        fetch(url).then(function (r) { return r.json(); }).then(function (payload) {
            window.dash_clientside.chart.applyLodWindow(payload);
        }).catch(function () {
            setLodLoading(false);
            delete lodResolvers[mySeq];
        });
    }

    function requestLodWindowAsync(tier, from, to, span) {
        return new Promise(function (resolve) { requestLodWindow(tier, from, to, span, resolve); });
    }

    // מבקש רק את העסקאות שחופפות לטווח [from, to] - נתיב Flask ישיר (/api/markers,
    // ראו app.py), לא Dash callback, מאותה סיבה בדיוק כמו /api/lod. seq פוסל
    // תשובות מיושנות (בדיוק כמו lodRequestSeq) - חשוב כאן במיוחד כי המשתמש יכול
    // לגלול/לזום מהר יותר ממה שסבב הלוך-חזור אחד לוקח.
    function requestMarkersWindow(from, to) {
        if (!markersAvailable || !currentSymbol || !currentTimeframe) return;
        if (from == null || to == null) return;
        var mySeq = ++markersRequestSeq;
        var url = "/api/markers?symbol=" + encodeURIComponent(currentSymbol) +
            "&timeframe=" + encodeURIComponent(currentTimeframe) +
            "&from=" + Math.round(from) + "&to=" + Math.round(to) +
            "&seq=" + mySeq;
        fetch(url).then(function (r) { return r.json(); }).then(function (payload) {
            if (Number(payload.seq) !== mySeq) return;  // תשובה מיושנת - בקשה חדשה יותר כבר יצאה
            if (!candleSeries) return;
            applyMarkers(candleSeries, payload.markers || []);
            var noteEl = document.getElementById("marker-note");
            if (noteEl) {
                if (payload.note) { noteEl.textContent = payload.note; noteEl.style.display = "block"; }
                else { noteEl.style.display = "none"; }
            }
        }).catch(function () {});
    }

    // רץ (עם debounce) בכל שינוי זום/גלילה. חלון buffer סביב הטווח הנראה (לא רק
    // הטווח עצמו) כדי שגלילה קטנה בתוך ה-buffer לא תדרוש בקשה נוספת מיידית.
    function onVisibleTimeRangeChanged(range) {
        if (!range || range.from == null || range.to == null) return;
        // תמיד מעדכנים, גם אם שאר הפונקציה יוצאת מוקדם למטה - נקרא ב-run-btn
        // (Add to chart) כדי להריץ בקטסט רק על מה שבאמת נראה עכשיו בגרף, לא
        // על כל ההיסטוריה. אותה מוסכמת epoch בדיוק כמו center ב-requestLodWindow.
        window.__currentVisibleRange = {from: range.from, to: range.to};
        if (!chart || !currentSymbol || !currentTimeframe) return;
        var span = range.to - range.from;
        if (span <= 0) return;

        // סימוני-עסקאות: בניגוד ל-LOD (למטה), מתעדכנים בכל שינוי טווח נראה, לא
        // רק כשחוצים את גבולות מה שכבר טעון - "עסקאות 2019" ו"עסקאות 2023" הן
        // תת-קבוצות שונות לגמרי גם באותה רזולוציית נרות בדיוק. debounce קצר
        // (לא הבלימה המורכבת של LOD) מספיק כי הבקשה עצמה זולה - סינון על 156K
        // עסקאות בפייתון הוא מילישניות, לא שאיבה מ-IB/דיסק.
        if (markersAvailable) {
            clearTimeout(markersDebounceTimer);
            markersDebounceTimer = setTimeout(function () {
                requestMarkersWindow(range.from, range.to);
            }, 150);
        }
        var tier = idealTierFor(span);
        var needFetch;
        if (tier !== viewTier || viewWindow.from == null || viewWindow.to == null) {
            needFetch = true;
        } else {
            // תואם ל-buffer בשרת (load_lod_window: פי 3 הטווח מכל צד) - שולפים
            // הרבה לפני שמגיעים בפועל לקצה מה שכבר טעון, כדי שגלילה רציפה
            // באותו כיוון תמשיך על דאטה שכבר קיים בלי לחכות לסבב נוסף כמעט אף פעם.
            var margin = span * 1.5;
            needFetch = (range.from - margin < viewWindow.from) || (range.to + margin > viewWindow.to);
        }
        if (!needFetch) return;
        // ליד קצה אמיתי (הדאטה ההיסטורית נגמרת, או שאנחנו ב"עכשיו" ואין עתיד לשלוף) -
        // margin לא באמת ניתן לסיפוק לעולם, אז needFetch יוצא true בכל טיק לייב עד
        // אינסוף, גם עם shiftVisibleRangeOnNewBar כבוי (הבר האחרון עדיין מתקדם ב-60
        // שניות בכל דקה, וזה משנה את range.to גם בלי שהמשתמש נגע בכלום). כל בקשה
        // כזו מקבלת seq גבוה יותר מכל גלילה אחורה אמיתית שקרתה קודם, ודורסת אותה
        // כש"המנצחת" - זה בדיוק מה שגרם לגרף "לקפוץ בחזרה" לחלון האחרון גם אחרי
        // גלילה אמיתית אחורה. פותרים בלימת בקשות שמרכזן לא זז משמעותית מהבקשה
        // הקודמת (לא רק קרוב לחלון הטעון, אלא ממש לא התקדם) - זה תופס את שני הקצוות
        // (עכשיו וגם תחילת ההיסטוריה) בלי לצטרך לדעת את "עכשיו" בפועל.
        var center = Math.round((range.from + range.to) / 2);
        if (lastRequestedCenter !== null && Math.abs(center - lastRequestedCenter) < span * 0.1) return;
        lastRequestedCenter = center;
        requestLodWindow(tier, range.from, range.to, span);
    }

    // ---------- "Go to date" (כפתור + מודל, פינה ימנית-תחתונה של הגרף) ----------
    // רק setVisibleRange - שינוי הטווח הנראה מפעיל בעצמו את onVisibleTimeRangeChanged
    // למעלה, בדיוק כמו גלילה/זום ידניים, אז אין צורך בשום לוגיקת טעינה נפרדת כאן.
    // תאריך/שעה שהמשתמש מקליד מתפרשים כשעון הבורסה (כמו כל שעה אחרת באפליקציה),
    // ומומרים לאותה מוסכמת "UTC מזויף" שהנרות עצמם משתמשים בה.
    function fakeUtcEpochFromLocal(dateStr, timeStr) {
        if (!dateStr) return null;
        var d = dateStr.split("-").map(Number);
        var t = (timeStr || "00:00").split(":").map(Number);
        if (d.length < 3 || isNaN(d[0]) || isNaN(d[1]) || isNaN(d[2])) return null;
        var ms = Date.UTC(d[0], d[1] - 1, d[2], t[0] || 0, t[1] || 0, 0);
        return Math.floor(ms / 1000);
    }

    function todayDisplayDateString() {
        // ה"היום" של המודל חייב להיות בשעון ישראל, לא בשעון הבורסה: הגרף עצמו
        // מציג את הנרות בשעון ישראל (to_display_naive/_epoch ב-data.py), אז ברירת
        // המחדל כאן חייבת להתאים לאותו שעון — אחרת לפעמים "היום" של ניו-יורק היה
        // כבר אתמול/מחר ביחס למה שבאמת מוצג על הגרף.
        // en-CA מחזיר ישירות בפורמט YYYY-MM-DD, בדיוק מה ש-<input type=date> מצפה לו.
        return new Intl.DateTimeFormat("en-CA", {timeZone: "Asia/Jerusalem"}).format(new Date());
    }

    function openGotoModal() {
        var overlay = document.getElementById("goto-modal-overlay");
        if (!overlay) return;
        var today = todayDisplayDateString();
        [["goto-date", today], ["goto-range-from-date", today], ["goto-range-to-date", today]]
            .forEach(function (pair) {
                var el = document.getElementById(pair[0]);
                if (el && !el.value) el.value = pair[1];
            });
        overlay.classList.add("open");
    }

    function closeGotoModal() {
        var overlay = document.getElementById("goto-modal-overlay");
        if (overlay) overlay.classList.remove("open");
    }

    function switchGotoTab(tab) {
        var isDate = tab === "date";
        document.getElementById("goto-tab-date").classList.toggle("goto-tab-active", isDate);
        document.getElementById("goto-tab-range").classList.toggle("goto-tab-active", !isDate);
        document.getElementById("goto-pane-date").style.display = isDate ? "block" : "none";
        document.getElementById("goto-pane-range").style.display = isDate ? "none" : "block";
    }

    // מוצא את אינדקס הבר הקרוב ביותר לזמן נתון בתוך allCandles (ממוין עולה,
    // חיפוש בינארי) - ראו למה למטה, בסוף submitGoto.
    function closestCandleIndex(targetEpoch) {
        var n = allCandles.length;
        if (!n) return null;
        if (targetEpoch <= allCandles[0].time) return 0;
        if (targetEpoch >= allCandles[n - 1].time) return n - 1;
        var lo = 0, hi = n - 1;
        while (lo < hi) {
            var mid = (lo + hi) >> 1;
            if (allCandles[mid].time < targetEpoch) lo = mid + 1; else hi = mid;
        }
        if (lo > 0 && Math.abs(allCandles[lo - 1].time - targetEpoch) <= Math.abs(allCandles[lo].time - targetEpoch)) {
            return lo - 1;
        }
        return lo;
    }

    // טוען קודם את הדאטה לתאריך היעד, ורק אז מזיז את הטווח הנראה אליו - לא
    // סומכים על setVisibleRange לטווח שאין לו חפיפה עם מה שכבר טעון שיפעיל
    // בעצמו את subscribeVisibleTimeRangeChange (זה לא תמיד קורה: setVisibleRange
    // לטווח "זר" לחלוטין הוא בדיוק המקרה שבו נצפתה תקיעות/מסך ריק). כך בזמן
    // ש-setVisibleRange נקרא, הדאטה לתאריך כבר שם - אותה מוסכמה בדיוק כמו טעינה
    // ראשונית רגילה.
    async function submitGoto() {
        if (!chart) { closeGotoModal(); return; }
        var dateTabActive = document.getElementById("goto-tab-date").classList.contains("goto-tab-active");
        var centerFrom, centerTo;

        if (dateTabActive) {
            var target = fakeUtcEpochFromLocal(
                document.getElementById("goto-date").value,
                document.getElementById("goto-time").value
            );
            if (target === null) return;
            // שומרים על רוחב-הזום הנוכחי, רק מזיזים את החלון להיות ממורכז סביב התאריך.
            var vr = chart.timeScale().getVisibleRange();
            var span = (vr && vr.to != null && vr.from != null) ? (vr.to - vr.from) : (7 * 24 * 3600);
            centerFrom = target - span / 2;
            centerTo = target + span / 2;
        } else {
            var from = fakeUtcEpochFromLocal(
                document.getElementById("goto-range-from-date").value,
                document.getElementById("goto-range-from-time").value
            );
            var to = fakeUtcEpochFromLocal(
                document.getElementById("goto-range-to-date").value,
                document.getElementById("goto-range-to-time").value
            );
            if (from === null || to === null || from >= to) return;
            centerFrom = from;
            centerTo = to;
        }

        var submitBtn = document.getElementById("goto-submit");
        var originalLabel = submitBtn ? submitBtn.textContent : null;
        if (submitBtn) { submitBtn.disabled = true; submitBtn.textContent = "טוען..."; }

        var span2 = centerTo - centerFrom;
        await requestLodWindowAsync(idealTierFor(span2), centerFrom, centerTo, span2);

        if (submitBtn) { submitBtn.disabled = false; submitBtn.textContent = originalLabel; }

        // setVisibleRange (מבוסס-זמן) לא אמין כאן: אחרי requestLodWindowAsync,
        // allCandles יכול להכיל גם את הדאטה שהיה נטען לפני הקפיצה וגם את דאטה
        // היעד החדש (applyLodWindow ממזג, לא מחליף, כשהרזולוציה זהה) - עם פער
        // עצום בזמן ביניהם (למשל 2020 ל-2026). Lightweight Charts ממפה זמן->
        // אינדקס-לוגי לפי מרחק-ברים *ממוצע* על פני כל הסדרה - פער כזה מעוות
        // את החישוב הזה ומייצר תצוגה שבורה לגמרי (נרות ענקיים/מעוותים, לא
        // בהכרח בתאריך המבוקש) - זה בדיוק מה שנצפה בבדיקה. setVisibleLogicalRange
        // על האינדקסים שבאמת הכי קרובים ל-centerFrom/centerTo בתוך allCandles
        // עוקף את זה לגמרי: מיקום ישיר לפי הבר בפועל, לא אינטרפולציה של זמן.
        var idxFrom = closestCandleIndex(centerFrom);
        var idxTo = closestCandleIndex(centerTo);
        if (idxFrom !== null && idxTo !== null && idxTo > idxFrom) {
            chart.timeScale().setVisibleLogicalRange({from: idxFrom, to: idxTo});
        } else {
            chart.timeScale().setVisibleRange({from: centerFrom, to: centerTo});
        }
        closeGotoModal();
    }

    // ---------- שעון בזמן אמת (סרגל תחתון, #market-clock) ----------
    // JS טהור בצד לקוח בלבד - setInterval כל 1000 מ"ש, בלי שום תלות בשרת
    // (בניגוד ל-secondsToClose/marketOpen, שמגיעים מ-payload.updateLive).
    // toLocaleTimeString עם timeZone: "Asia/Jerusalem" מפעיל את כללי הקיץ/חורף
    // האמיתיים של ישראל דרך הדפדפן עצמו - אין כאן שום +2/+3 מקודד בקוד.
    (function () {
        var clockEl = null;
        function tick() {
            if (!clockEl) clockEl = document.getElementById("market-clock");
            if (!clockEl) return;
            clockEl.textContent = new Date().toLocaleTimeString("he-IL", {timeZone: "Asia/Jerusalem"});
        }
        tick();
        setInterval(tick, 1000);
    })();

    document.addEventListener("click", function (e) {
        if (e.target.closest("#reset-live-btn")) {
            if (chart) chart.timeScale().scrollToRealTime();
            return;
        }
        if (e.target.closest("#goto-btn")) { openGotoModal(); return; }
        if (e.target.closest("#goto-close") || e.target.closest("#goto-cancel")) { closeGotoModal(); return; }
        if (e.target.id === "goto-modal-overlay") { closeGotoModal(); return; }
        if (e.target.closest("#goto-tab-date")) { switchGotoTab("date"); return; }
        if (e.target.closest("#goto-tab-range")) { switchGotoTab("range"); return; }
        if (e.target.closest("#goto-submit")) { submitGoto(); return; }
    });
    document.addEventListener("keydown", function (e) {
        if (e.key === "Escape") closeGotoModal();
    });

    window.dash_clientside = window.dash_clientside || {};
    window.dash_clientside.chart = {
        render: function (payload) {
            var el = document.getElementById("chart-container");
            if (!el || !payload || !payload.candles) return window.dash_clientside.no_update;
            if (typeof LightweightCharts === "undefined") return window.dash_clientside.no_update;

            ensureChart(el, payload);

            var key = payload.symbol + "|" + payload.timeframe;
            var isNewKey = key !== lastKey;
            if (isNewKey) {
                // סימבול/טיים-פריים חדש לגמרי: מחליפים את כל הדאטה המקומי, לא ממזגים.
                // מתחילים תמיד ברזולוציה המלאה (viewTier = הטיים-פריים שנבחר) - בדיוק
                // כמו טעינה ראשונית תמיד הייתה; מדרג-הפירוט ייכנס לפעולה רק אם/כש
                // המשתמש יזום/יגלול לטווח שכבר לא הגיוני להראות ברזולוציה הזו.
                allCandles = payload.candles;
                allVolume = payload.volume.map(decodeVolume);
                viewTier = payload.timeframe;
                lodRequestSeq++;  // מבטל כל בקשת LOD ישנה שעוד "בדרך" מהסימבול/טיים-פריים הקודם
                lastRequestedCenter = null;  // סימבול/טיים-פריים חדש - לא לבלום מול בקשה של תצוגה קודמת
                markersAvailable = false;  // תוצאת בקטסט קודמת לא רלוונטית לסימבול/טיים-פריים החדש
                var noteElReset = document.getElementById("marker-note");
                if (noteElReset) noteElReset.style.display = "none";
            }
            candleSeries.setData(allCandles);
            volumeSeries.setData(allVolume);
            applyMarkers(candleSeries, payload.markers || []);  // תמיד [] מהשרת - מנקה סימונים ישנים מיד
            closeHistory = allCandles.map(function (c) { return {time: c.time, close: c.close}; });
            smaSeries.setData(computeFullSMA(closeHistory, SMA_PERIOD));
            viewWindow = {
                from: allCandles.length ? allCandles[0].time : null,
                to: allCandles.length ? allCandles[allCandles.length - 1].time : null,
            };
            var lastCandle = allCandles.length ? allCandles[allCandles.length - 1] : null;
            lastBarTime = lastCandle ? lastCandle.time : null;
            lastClose = lastCandle ? lastCandle.close : null;
            lastOpen = lastCandle ? lastCandle.open : null;
            currentTimeframe = payload.timeframe;
            currentSymbol = payload.symbol;

            // בקטסט חדש הסתיים בהצלחה (ראו app.py: run_or_switch) - מבקשים מיד
            // את סימוני-העסקאות לטווח הנראה הנוכחי, בלי לחכות לגלילה/זום הבאים
            // (onVisibleTimeRangeChanged ידאג לעדכן אותם משם והלאה).
            if (payload.backtestReady) {
                markersAvailable = true;
                var vr = window.__currentVisibleRange || viewWindow;
                requestMarkersWindow(vr.from, vr.to);
            }

            // מאפסים את התצוגה (זום) רק כשבאמת עברנו לסימבול/טיים-פריים אחר. הרצת
            // בקטסט מוסיפה סימונים בלבד ולא אמורה לזרוק את המשתמש מהמקום שהוא
            // הסתכל עליו.
            if (isNewKey) {
                pendingFit = true;
                fitDeadline = Date.now() + 1200;
                maybeFit();
                lastKey = key;
                // ברירת מחדל לפני שהמשתמש בכלל גלל/זום - כל מה שכבר נטען,
                // כדי ש-run-btn לא ייתקל ב-undefined אם לוחצים מיד אחרי טעינה.
                window.__currentVisibleRange = {from: viewWindow.from, to: viewWindow.to};
            }
            positionCountdown();
            return "";
        },

        // מדרג-פירוט: תשובה לבקשת חלון (requestLodWindow למעלה) - מחליפה את מה
        // שמצויר, לא ממזגת אליו (יכולה להיות רזולוציה שונה לגמרי). setData() לא
        // מזיז לבד את הטווח הנראה כשהזמנים חופפים למה שהיה, אז הזום/גלילה נשארים
        // בדיוק כמו שהיו - בדיוק בשביל זה requestLodWindow מבקש buffer סביב מה
        // שבאמת נראה, לא רק את זה.
        applyLodWindow: function (payload) {
            // "Go to date" (requestLodWindowAsync) מחכה לתוצאה הזו כדי לדעת מתי
            // בטוח להזיז את הטווח הנראה - נפתר תמיד, בכל נתיב יציאה. resolve
            // נשלף לפי ה-seq של התשובה הזו בדיוק (לא משתנה גלובלי משותף) - כדי
            // שבקשה מתחרה (הגלילה התגובתית, שרצה ברקע כל הזמן) לא תדרוס ותתקע
            // הבטחה של בקשה אחרת שעוד ממתינה לתשובה שלה.
            var seqKey = payload && payload.seq;
            var resolve = seqKey != null ? lodResolvers[seqKey] : null;
            if (seqKey != null) delete lodResolvers[seqKey];
            var finish = function (ok) { setLodLoading(false); if (resolve) resolve(ok); };

            if (!payload || !payload.candles) { finish(false); return ""; }
            if (payload.symbol !== currentSymbol) { finish(false); return ""; }   // כבר עברו לסימבול אחר
            if (Number(payload.seq) !== lodRequestSeq) { finish(false); return ""; }  // תשובה מיושנת - Number() כי /api/lod (query string) עלול להחזיר seq כמחרוזת
            if (!payload.candles.length) { finish(false); return ""; }             // אין דאטה בחלון הזה - משאירים מה שיש
            if (!chart || !candleSeries || !volumeSeries) { finish(false); return ""; }

            if (payload.tier === viewTier && allCandles.length) {
                // אותה רזולוציה כמו מה שכבר מוצג - ממזגים לתוך מה שכבר טעון במקום
                // להחליף. בלי זה, כל תשובת LOD (גם חלון שחופף כמעט לגמרי למה שכבר
                // היה) זרקה את כל הדאטה הקודם והשאירה רק את חלון ה-buffer האחרון -
                // בדיוק מה שגרם גם ל"קפיצה" חזותית בכל עדכון (setData מלא על טווח
                // שונה בכל פעם) וגם לזה שאף פעם לא רואים יותר מחלון-buffer בודד,
                // למרות שהקאש מכיל היסטוריה עמוקה בהרבה ממה שבאמת מצטבר על הגרף.
                var byTime = {};
                allCandles.forEach(function (c) { byTime[c.time] = c; });
                payload.candles.forEach(function (c) { byTime[c.time] = c; });
                var times = Object.keys(byTime).map(Number).sort(function (a, b) { return a - b; });
                allCandles = times.map(function (t) { return byTime[t]; });

                var volByTime = {};
                allVolume.forEach(function (v) { volByTime[v.time] = v; });
                payload.volume.map(decodeVolume).forEach(function (v) { volByTime[v.time] = v; });
                allVolume = times.map(function (t) { return volByTime[t]; }).filter(function (v) { return v; });
            } else {
                // רזולוציה שונה (זימום עבר סף טיים-פריים) - אי אפשר למזג בארים
                // בגדלים שונים לתוך אותה סדרה, מחליפים לגמרי כמו קודם.
                viewTier = payload.tier;
                allCandles = payload.candles;
                allVolume = payload.volume.map(decodeVolume);
            }
            viewWindow = {from: allCandles[0].time, to: allCandles[allCandles.length - 1].time};
            candleSeries.setData(allCandles);
            volumeSeries.setData(allVolume);
            closeHistory = allCandles.map(function (c) { return {time: c.time, close: c.close}; });
            smaSeries.setData(computeFullSMA(closeHistory, SMA_PERIOD));
            var lastCandle = allCandles[allCandles.length - 1];
            lastBarTime = lastCandle.time;
            lastClose = lastCandle.close;
            lastOpen = lastCandle.open;
            updateHeaderText();
            positionCountdown();
            finish(true);
            return "";
        },

        // עדכון לייב: מגיע כל כמה שניות מ-app.py עם הנר/ות האחרון/ים ברזולוציה
        // המלאה (currentTimeframe). אם הרזולוציה המוצגת כרגע (viewTier) גסה יותר
        // (זומם החוצה) - מצרפים את הבר העדין ל"סל" הגס האחרון (open/high/low/close/
        // volume) במקום להראות אותו כמות שהוא. series.update מעדכן את הנר הפתוח
        // אם הזמן זהה לאחרון שכבר מצויר, או מוסיף נר חדש אם הזמן התקדם.
        updateLive: function (payload) {
            if (!chart || !candleSeries || !volumeSeries) return window.dash_clientside.no_update;
            if (!payload) return window.dash_clientside.no_update;

            var key = payload.symbol + "|" + payload.timeframe;
            if (key !== lastKey) return "";  // המשתמש כבר עבר לגרף אחר — מתעלמים מהתגובה המאוחרת

            // הטיימר מוצג רק כשהשוק "חי" בפועל (payload.marketOpen, נקבע ב-app.py
            // לפי live.is_market_live - טיק אמיתי הגיע לאחרונה, לא חישוב שעון-לוח
            // קבוע). marketOpen===false (שוק סגור/TWS מנותק/אין עדיין טיק) - מאפסים
            // secondsLeft ל-null כדי ש-positionCountdown יסתיר אותו לגמרי (אותה
            // בדיקה בדיוק שכבר קיימת שם ל-lastClose===null). ברגע שטיקים חוזרים
            // marketOpen שוב true בעדכון הבא - הטיימר חוזר אוטומטית, בלי לוגיקה נוספת.
            if (payload.marketOpen === false) {
                secondsLeft = null;
                positionCountdown();
            } else if (typeof payload.secondsToClose === "number") {
                // מסתנכרנים עם שעון השרת (שעון הבורסה) רק כשיש סטייה אמיתית. תגובה
                // של Dash יכולה להתעכב כשהשרת עסוק בשאיבה ארוכה, ואז היא מגיעה
                // מיושנת — יישור עיוור אליה היה גורם לטיימר לקפוץ אחורה ולהיראות תקוע.
                if (secondsLeft === null || Math.abs(payload.secondsToClose - secondsLeft) > 2) {
                    secondsLeft = payload.secondsToClose;
                }
                positionCountdown();
            }
            // מטען ללא נרות מגיע כששוק סגור או שאין TWS — הטיימר עדיין מתעדכן.
            if (!payload.candles || !payload.candles.length) return "";

            var tierSec = TIER_SECONDS[viewTier] || BAR_SECONDS[currentTimeframe];

            for (var i = 0; i < payload.candles.length; i++) {
                var bar = payload.candles[i];
                var volRaw = decodeVolume(payload.volume[i]);
                // ה"סל" הגס (viewTier) שהבר העדין הזה שייך אליו - זהה ל-bar.time
                // עצמו כשאין זימום (viewTier == currentTimeframe, tierSec תואם).
                var bucketTime = Math.floor(bar.time / tierSec) * tierSec;
                if (lastBarTime !== null && bucketTime < lastBarTime) continue;
                // אם המשתמש רחוק בעבר (אחרי "Go to date" או זום להיסטוריה רחוקה,
                // ימים+) - הבר החי הזה לא שייך למה שמוצג כרגע, ואסור להדביק אותו
                // כ"נר חדש" רחוק בעתיד בקצה התצוגה. הסף חייב להיות רחב מאוד: הוא
                // נבדק גם מול lastBarTime הראשוני (הבר האחרון בשאיבה ההיסטורית
                // הרגילה) - שיכול באופן לגיטימי לפגר כמה דקות אחרי "עכשיו" האמיתי
                // (רענון קאש/pacing), ולא רק כשבאמת גוללים להיסטוריה רחוקה. סף
                // צר מדי (היה 5 יחידות-tier = 5 דקות ב-1m) חוסם כל עדכון לייב
                // רגיל לצמיתות - בדיוק מה שגרם ל"לייב לא זז בכלל".
                if (lastBarTime !== null && bucketTime - lastBarTime > tierSec * 1440) continue;

                var displayBar;
                if (lastBarTime === bucketTime && allCandles.length) {
                    // בר עדיין פתוח - מעדכנים רק מחיר (open/high/low/close) בזמן אמת,
                    // לא נוגעים בוולום בכלל כאן. bar.volume מהשרת הוא תמיד וולום-
                    // הבר-עד-עכשיו המלא (לא תוספת מאז העדכון הקודם - ראו live.py:
                    // _bar_from_ib), אז "topVal + volRaw.value" שהיה כאן קודם חיבר
                    // את הערך המלא הזה על גבי מה שכבר מוצג *בכל טיק* - עמודת הוולום
                    // תפחה עשרות מונים תוך דקה אחת. הוולום מתעדכן רק כשבר חדש נפתח
                    // (הענף "else" למטה) - אז מקבל ערך נכון וסופי, לא מצטבר.
                    var top = allCandles[allCandles.length - 1];
                    displayBar = {
                        time: bucketTime, open: top.open,
                        high: Math.max(top.high, bar.high), low: Math.min(top.low, bar.low),
                        close: bar.close,
                    };
                    allCandles[allCandles.length - 1] = displayBar;
                    candleSeries.update(displayBar);
                } else {
                    var displayVol = {time: bucketTime, value: volRaw.value, color: volRaw.color};
                    displayBar = {time: bucketTime, open: bar.open, high: bar.high, low: bar.low, close: bar.close};
                    allCandles.push(displayBar);
                    allVolume.push(displayVol);
                    candleSeries.update(displayBar);
                    volumeSeries.update(displayVol);
                }
                lastBarTime = bucketTime;
                lastClose = displayBar.close;
                lastOpen = displayBar.open;
                viewWindow.to = bucketTime;

                // מעדכנים את עותק ההיסטוריה המקומי (לחישוב SMA) באותה לוגיקת
                // "עדכון אם אותו זמן, הוספה אם זמן חדש" כמו candleSeries.update עצמו.
                if (closeHistory.length && closeHistory[closeHistory.length - 1].time === bucketTime) {
                    closeHistory[closeHistory.length - 1].close = displayBar.close;
                } else {
                    closeHistory.push({time: bucketTime, close: displayBar.close});
                }
                var smaPoint = lastSMAPoint(closeHistory, SMA_PERIOD);
                if (smaPoint && smaSeries) smaSeries.update(smaPoint);
            }
            currentTimeframe = payload.timeframe;
            positionCountdown();
            return "";
        }
    };
})();
