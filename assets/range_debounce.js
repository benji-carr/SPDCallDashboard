(function () {
    "use strict";

    function makeRangeDebouncer(storeId) {
        // Each graph owns its timer; activity on one never delays the other.
        let timer = null;
        return function (relayout) {
            const hasRange = relayout && (
                (Array.isArray(relayout["xaxis.range"]) && relayout["xaxis.range"].length >= 2) ||
                (relayout["xaxis.range[0]"] != null && relayout["xaxis.range[1]"] != null) ||
                relayout["xaxis.autorange"] === true
            );
            if (!hasRange) {
                return window.dash_clientside.no_update;
            }

            clearTimeout(timer);
            timer = setTimeout(function () {
                timer = null;
                window.dash_clientside.set_props(storeId, {data: relayout});
            }, 750);
            // Do not keep a callback/Promise pending while the chart is dragged.
            return window.dash_clientside.no_update;
        };
    }

    window.dash_clientside = Object.assign({}, window.dash_clientside, {
        range_debounce: {
            calls: makeRangeDebouncer("calls-daily-relayout-debounced-store"),
            crime: makeRangeDebouncer("crime-daily-relayout-debounced-store")
        }
    });
}());
