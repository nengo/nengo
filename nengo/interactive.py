from IPython.display import SVG

txt = """
<svg xmlns="http://www.w3.org/2000/svg" versioN="1.1" height="100" width="500" >
<line x0="0" y0="0" x1="10" y1="10" id="foo" style="stroke:rgb(255,0,0);stroke-width:2" />
<script type="application/ecmascript"><![CDATA[

    var %(prefix)s_run_state = "stopped";
    var %(prefix)s_time_marker = document.getElementById("%(prefix)s_time_marker");

    function %(prefix)s_click_start(evt) {
        console.log('start');
        %(prefix)s_run_state = "running";

        function animation_step () {
            if (%(prefix)s_run_state == "stopped") return;
            console.log('step');
            console.log('%(py_model_obj)s.sim_obj.run_steps(%(sim_steps_per_anim_step)s);');
            IPython.notebook.kernel.execute(
                '%(py_model_obj)s.sim_obj.run_steps(%(sim_steps_per_anim_step)s);',
                {
                    'execute_reply': function(content, metadata){
                        console.log('execute_reply:');
                        console.log(content);
                        console.log(metadata);
                        console.log(content.user_variables.val.data);
                        console.log(content.user_expressions.n_steps.data);
                        %(prefix)s_time_marker.setAttribute("cx", content.user_expressions.n_steps.data['text/plain']);
                        console.log('-> done execute_reply');
                    },
                    'output': function(msg_type, content, metadata){
                        console.log('out');
                        if (0) {
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
    return SVG(txt % {
        'prefix': model_name, 
        'py_model_obj': model_name + '.model',
        'period_ms': period_ms,
        'sim_steps_per_anim_step': 1,
    })


