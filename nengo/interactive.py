from IPython.display import SVG, display

txt = """
<svg xmlns="http://www.w3.org/2000/svg" versioN="1.1" height="100" width="500" >
<script type="application/ecmascript"><![CDATA[

    var %(prefix)s_run_state = "stopped";
    var %(prefix)s_time_marker = document.getElementById("%(prefix)s_time_marker");

    function %(prefix)s_click_start(evt) {
        console.log('start');
        %(prefix)s_run_state = "running";

        function animation_step () {
            if (%(prefix)s_run_state == "stopped") return;
            if (0) {
                console.log('step');
                console.log('%(py_model_obj)s.sim_obj.run_steps(%(sim_steps_per_anim_step)s);');
            }
            IPython.notebook.kernel.execute(
                '%(py_model_obj)s.sim_obj.run_steps(%(sim_steps_per_anim_step)s);',
                {
                    'execute_reply': function(content, metadata){
                        if (0) {
                            console.log('execute_reply:');
                            console.log(content);
                            console.log(metadata);
                            console.log(content.user_variables.val.data);
                            console.log(content.user_expressions.n_steps.data);
                            console.log('-> done execute_reply');
                        }
                        %(prefix)s_time_marker.setAttribute("cx", content.user_expressions.n_steps.data['text/plain']);
                    },
                    'output': function(msg_type, content, metadata){
                        if (0) {
                            console.log('out');
                            console.log(msg_type);
                            console.log(content);
                            console.log(metadata);
                            console.log('-> done');
                        }
                    },
                    'clear_output': function(a){
                        console.log('co');
                    },
                    'set_next_input': function(a){
                        console.log('sni');
                    },
                },
                {
                    'silent': false,
                    'user_variables': ['val'],
                    'user_expressions': {
                        'n_steps': '%(py_model_obj)s.sim_obj.n_steps',
                        },
                });

            setTimeout(animation_step, %(period_ms)s);
        }
        animation_step();
    }
    function %(prefix)s_click_stop(evt) {
        console.log('stop');
        %(prefix)s_run_state = "stopped";
    }
    function %(prefix)s_click_reset(evt) {
        console.log('reset');
        %(prefix)s_run_state = "stopped";
        IPython.notebook.kernel.execute(
            '%(py_model_obj)s.sim_obj.reset();'
            );
        %(prefix)s_time_marker.setAttribute("cx", 0);
    }
    ]]> </script>

<!-- Act on each click event -->

<g onclick="%(prefix)s_click_start(evt)" >
  <rect x="0" y="0" width="100" height="30" fill="green">
  </rect>
  <text x="20" y="20" >Start</text>
  </g>

<g onclick="%(prefix)s_click_stop(evt)" >
  <rect x="200" cy="0" width="100" height="30" fill="red"></rect>
  <text x="220" y="20" >Stop</text>
</g>

<g onclick="%(prefix)s_click_reset(evt)" >
  <rect x="400" cy="0" width="100" height="30" fill="rgb(200, 200, 200)" ></rect>
  <text x="420" y="20" >Reset</text>
</g>

<circle cx="0" cy="50" r="10" fill="black" id="%(prefix)s_time_marker"></circle>

</svg>
"""


def old_api_controller(model, model_name, period_ms=100):
    if getattr(model.model, 'sim_obj', None) is None:
        model.model.sim_obj = model.model.simulator(model.model)
    display(SVG(txt % {
        'prefix': model_name, 
        'py_model_obj': model_name + '.model',
        'period_ms': period_ms,
        'sim_steps_per_anim_step': 1,
    }))




probe_2d_txt = """
<svg xmlns="http://www.w3.org/2000/svg" versioN="1.1" height="%(widget_height)s" width="%(widget_width)s" >
<script type="application/ecmascript"><![CDATA[

    var %(probe_name)s_old_x = [0, 0, 0];
    var %(probe_name)s_old_y = [0, 0, 0];

    function %(probe_name)s_update() {
        var %(probe_name)s_dot0 = document.getElementById("%(probe_name)s_dot0");
        var %(probe_name)s_dot1 = document.getElementById("%(probe_name)s_dot1");
        var %(probe_name)s_dot2 = document.getElementById("%(probe_name)s_dot2");
        if (%(prefix)s_run_state == "running" )
        {
            var expr = 'list(%(py_model_obj)s.sim_obj.probe_outputs[%(probe_name)s.probe][-1])';
            IPython.notebook.kernel.execute(
                'pass',
                {
                    'execute_reply': function(content, metadata){
                        console.log('execute_reply:');
                        //console.log(metadata);
                        //console.log(content.user_expressions.probe_data);
                        //console.log(content.user_expressions.probe_data.data);
                        var pair = JSON.parse(content.user_expressions.probe_data.data['text/plain']);
                        var cx = (pair[0] + 1.5) / 3.0 * %(widget_width)s;
                        var cy = (pair[1] + 1.5) / 3.0 * %(widget_height)s;
                        %(probe_name)s_old_x[2] = %(probe_name)s_old_x[1];
                        %(probe_name)s_old_x[1] = %(probe_name)s_old_x[0];
                        %(probe_name)s_old_x[0] = cx;
                        %(probe_name)s_old_y[2] = %(probe_name)s_old_y[1];
                        %(probe_name)s_old_y[1] = %(probe_name)s_old_y[0];
                        %(probe_name)s_old_y[0] = cy;
                        //console.log(pair);
                        //console.log(cx);
                        //console.log(cy);
                        %(probe_name)s_dot0.setAttribute("cx", cx);
                        %(probe_name)s_dot0.setAttribute("cy", cy);
                        %(probe_name)s_dot1.setAttribute("cx",
                            %(probe_name)s_old_x[1]);
                        %(probe_name)s_dot1.setAttribute("cy",
                            %(probe_name)s_old_y[1]);
                        %(probe_name)s_dot2.setAttribute("cx",
                            %(probe_name)s_old_x[2]);
                        %(probe_name)s_dot2.setAttribute("cy",
                            %(probe_name)s_old_y[2]);
                        //console.log('-> done execute_reply');
                    },
                    'output': function(msg_type, content, metadata){
                        console.log('out');
                        if (1) {
                            console.log(msg_type);
                            console.log(content);
                            console.log(metadata);
                            console.log('-> done');
                        }
                    },
                    'clear_output': function(a){
                        console.log('co');
                    },
                    'set_next_input': function(a){
                        console.log('sni');
                    },
                },
                {
                    'silent': false,
                    'user_expressions': { 'probe_data': expr, },
                }
            );
        }
        setTimeout(%(probe_name)s_update, %(period_ms)s);
    }
]]> </script>

<rect x="0" cy="0" width="%(rect_width)s" height="%(rect_height)s"
    fill="white"
    stroke="black"
    stroke-width="1"
></rect>
<circle
    onload="%(probe_name)s_update()"
    id="%(probe_name)s_dot2"
    cx="0"
    cy="0"
    r="5"
    fill="rgb(200, 200, 200)"
></circle>
<circle
    onload="%(probe_name)s_update()"
    id="%(probe_name)s_dot1"
    cx="0"
    cy="0"
    r="5"
    fill="rgb(100, 100, 100)"
></circle>
<circle
    onload="%(probe_name)s_update()"
    id="%(probe_name)s_dot0"
    cx="0"
    cy="0"
    r="5"
    fill="black"
></circle>
</svg>
"""

def probe_2d(model_name, probe_name, width=250, height=250, period_ms=100):
    rendered = probe_2d_txt % {
        'prefix': model_name, 
        'py_model_obj': model_name + '.model',
        'period_ms': period_ms,
        'sim_steps_per_anim_step': 1,
        'widget_width': width,
        'widget_height': height,
        'rect_width': width - 2,
        'rect_height': height - 2,
        'probe_name': probe_name,
    }
    # print rendered
    display(SVG(rendered))


