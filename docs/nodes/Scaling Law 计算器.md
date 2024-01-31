
<script src="https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js"></script>

<script type="text/javascript" src="https://cdn.bokeh.org/bokeh/release/bokeh-3.3.3.js"></script>
<script type="text/javascript" src="https://cdn.bokeh.org/bokeh/release/bokeh-widgets-3.3.3.min.js"></script>
<script type="text/javascript" src="https://cdn.bokeh.org/bokeh/release/bokeh-tables-3.3.3.min.js"></script>
<script type="text/javascript" src="https://cdn.jsdelivr.net/npm/@holoviz/panel@1.3.8/dist/panel.min.js"></script>

## FLOPs计算器

<div id="flop_calc"></div>
<script type="text/javascript">
    async function main(){
    let pyodide = await loadPyodide();
    await pyodide.loadPackage("micropip");
    const micropip = pyodide.pyimport("micropip");
    await micropip.install([
        "https://cdn.holoviz.org/panel/1.3.8/dist/wheels/bokeh-3.3.3-py3-none-any.whl",
        "https://cdn.holoviz.org/panel/1.3.8/dist/wheels/panel-1.3.8-py3-none-any.whl"]
    )
    pyodide.runPython(`
        import panel as pn
        from bokeh.models.formatters import PrintfTickFormatter

        pn.extension(sizing_mode="stretch_width")

        ModelSelector = pn.widgets.Select(name="model", value="LLaMa2 7B", options=[
            "Custom Model",
            "LLaMa 7B",
            "LLaMa2 7B",
        ])

        NSlider = pn.widgets.FloatSlider(start=1, end=200, step=0.1, 
            format=PrintfTickFormatter(format='%.1f B'), name='模型规模（B）')
        DSlider = pn.widgets.FloatSlider(start=1, end=5000, step=100, 
            format=PrintfTickFormatter(format='%.1f B'), name='数据规模（B）')
        CText = pn.widgets.StaticText(name="算力需求C（pf-day）")
        pfday = pn.widgets.StaticText(value="pf-day为1P FLOPs x 24小时，约为8.64E19")

        def callback(new):
            ModelSelector.value = "Custom Model"
            CText.value = "%.1f pf-day"%(6*NSlider.value*1E9*DSlider.value*1E9/8.64E19)
            return

        def select_model(new):
            if new == "Custom Model":
                return
            if new == "LLaMa2 7B":
                NSlider.value=7.0
                DSlider.value=2000
            elif new == "LLaMa 7B":
                NSlider.value=7.0
                DSlider.value=1500


        pn.Column(
            "算力计算器",
            ModelSelector,
            NSlider, DSlider, 
            CText, pfday,
            pn.bind(callback, NSlider),
            pn.bind(callback, DSlider),
            pn.bind(select_model, ModelSelector),
        ).servable(target='flop_calc');
    `);
    }
    main();
</script>