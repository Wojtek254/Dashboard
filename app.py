                    c1.metric("Min (area)", f"{user_min:.4f}")
                    c2.metric("Max (area)", f"{user_max:.4f}")
                    c3.metric("Mean (area)", f"{user_mean:.4f}")
                    c4.metric("In-range pixels", f"{pixel_count_inrange}")
                    c5.metric("Total valid pixels", f"{pixel_count_total}")

                if region_ts:
                    df_r = pd.DataFrame(region_ts)
                    df_r = df_r.sort_values("date").set_index("date")

                    if left_aggregation == "count":
                        st.caption("The chart below shows daily spatial coverage; the map counts valid days per pixel across the full period.")
                        plot_pixelcount_timeseries(
                            df_r, f"Daily pixel coverage in MAIN area ({cygnss_layer_label(left_cygnss_layer)})"
                        )
                    else:
                        st.caption("The charts below use individual daily images; the summary above describes the aggregated map.")
                        col_ts, col_cnt = st.columns(2)

                        with col_ts:
                            title_ts = (
                                f"Daily min / max / mean anomaly (MAIN area, {cygnss_layer_label(left_cygnss_layer)})"
                                if stats_kind == "anomaly"
                                else f"Daily min / max / mean (MAIN area, {cygnss_layer_label(left_cygnss_layer)})"
                            )
                            plot_timeseries(df_r, title_ts, stats_kind, left_thr_max)

                        with col_cnt:
                            title_cnt = f"Daily pixel counts in MAIN area ({cygnss_layer_label(left_cygnss_layer)})"
                            plot_pixelcount_timeseries(df_r, title_cnt)
                else:
                    st.info("No data available to draw MAIN time series for the selected area after masking.")
        else:
            st.info("Draw a rectangular area on the map using the drawing tool.")
    else:
        st.info("Draw a rectangular area on the map using the drawing tool (rectangle icon in the top-left corner).")

st.markdown("---")
st.subheader("Export")
st.caption(
    "Download the analysis report for dates, rectangle coordinates, configuration, and statistics. "
    "For a visual copy of the complete dashboard, use the screenshot button and select this browser tab."
)

view_info = {
    "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    "left_date_range": (
        f"{left_start_date.strftime('%Y-%m-%d')} to {left_end_date.strftime('%Y-%m-%d')}"
    ),
    "left_days_used": len(left_sel_days),
    "left_aggregation": AGGREGATIONS[left_aggregation] if is_temporal_layer(left_shading_layer) else "N/A",
    "left_temporal_mode": (
        f"{left_window_days}-day window ending on selected date" if left_temporal_mode == "Rolling window" else "date range"
    ),
    "right_date_range": (
        f"{right_start_date.strftime('%Y-%m-%d')} to {right_end_date.strftime('%Y-%m-%d')} "
        f"({len(right_sel_days)} data days used)"
        if split_view and right_start_date is not None
        else "Split view disabled"
    ),
    "map_center": st.session_state.map_center,
    "right_aggregation": (
        AGGREGATIONS[right_aggregation]
        if split_view and is_temporal_layer(right_shading_layer) else "N/A"
    ),
    "right_temporal_mode": (
        f"{right_window_days}-day window ending on selected date" if right_temporal_mode == "Rolling window" else "date range"
    ) if split_view else "N/A",
    "map_zoom": st.session_state.map_zoom,
    "left_shading": LAYER_OPTIONS[left_shading_layer],
    "left_auxiliary": LAYER_OPTIONS[left_auxiliary_layer],
    "left_auxiliary_aggregation": (
        AGGREGATIONS[left_auxiliary_aggregation]
        if left_auxiliary_layer in TEMPORAL_EXTERNAL_LAYERS else "N/A"
    ),
    "left_auxiliary_scale": (
        f"{left_auxiliary_range[0]:g} to {left_auxiliary_range[1]:g} {OVERLAY_LEGENDS[left_auxiliary_layer]['unit']}".strip()
        if left_auxiliary_range is not None else "N/A"
    ),
    "right_shading": LAYER_OPTIONS[right_shading_layer] if split_view else "None",
    "right_auxiliary": LAYER_OPTIONS[right_auxiliary_layer] if split_view else "None",
    "right_auxiliary_aggregation": (
        AGGREGATIONS[right_auxiliary_aggregation]
        if split_view and right_auxiliary_layer in TEMPORAL_EXTERNAL_LAYERS else "N/A"
    ),
    "right_auxiliary_scale": (
        f"{right_auxiliary_range[0]:g} to {right_auxiliary_range[1]:g} {OVERLAY_LEGENDS[right_auxiliary_layer]['unit']}".strip()
        if right_auxiliary_range is not None else "N/A"
    ),
}

xmin_report, ymin_report, xmax_report, ymax_report = (
    region_bounds if region_bounds is not None else (None, None, None, None)
)

stats_info = {
    "source": (
        f"MAIN {LAYER_OPTIONS[left_cygnss_layer]}"
        if left_cygnss_layer is not None
        else "N/A"
    ),
    "date_range": view_info["left_date_range"],
    "thr_min": left_thr_min if left_thr_min is not None else "N/A",
    "thr_max": left_thr_max if left_thr_max is not None else "N/A",
    "unit": "N/A" if left_cygnss_layer is None else ("days" if left_aggregation == "count" else "%"),
    "region_drawn": region_drawn,
    "west_lon": f"{xmin_report:.6f} deg" if xmin_report is not None else "N/A",
    "east_lon": f"{xmax_report:.6f} deg" if xmax_report is not None else "N/A",
    "south_lat": f"{ymin_report:.6f} deg" if ymin_report is not None else "N/A",
    "north_lat": f"{ymax_report:.6f} deg" if ymax_report is not None else "N/A",
    "min": f"{user_min:.4f}" if user_min is not None else "N/A",
    "max": f"{user_max:.4f}" if user_max is not None else "N/A",
    "mean": f"{user_mean:.4f}" if user_mean is not None else "N/A",
    "count_inrange": str(pixel_count_inrange) if pixel_count_inrange is not None else "N/A",
    "count_total": str(pixel_count_total) if pixel_count_total is not None else "N/A",
}

png_bytes = build_png_report(view_info=view_info, stats_info=stats_info)
st.download_button(
    label="Download analysis report PNG",
    data=png_bytes,
    file_name=f"cygnss_snapshot_{dt.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png",
    mime="image/png",
)

render_fullpage_screenshot_button()
