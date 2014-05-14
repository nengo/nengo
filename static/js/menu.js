//*****************
// Setup the menu
//*****************

var server_last_modified_time = null;

//Load the browser and hide it
function open_file(file) {
    $('#filebrowser').hide();

    $('#filename').val(file)

    var data = new FormData();
    data.append('filename', file);

    var xhr = new XMLHttpRequest();
    xhr.open('POST', '/openfile', true);
    xhr.onload = function (event) {
        r = JSON.parse(this.responseText);
        editor.setValue(r.text);
        server_last_modified_time = r.mtime;
        $('#menu_save').addClass('disable');
    };
    xhr.send(data);
}

function save_file() {
    if ($('#menu_save').hasClass('disable')) {
        return;
    }
    $('#filebrowser').hide();

    var data = new FormData();
    data.append('filename', $('#filename').val());
    data.append('code', editor.getValue());

    var xhr = new XMLHttpRequest();
    xhr.open('POST', '/savefile', true);
    xhr.onload = function (event) {

        if (this.responseText == "success\n") {
            $('#menu_save').addClass('disable');
        } else {
            alert('Save error: ' + this.responseText);
        }
    };
    xhr.send(data);
}

function edit_filename() {
    $('#menu_save').removeClass('disable');

    fn = $('#filename');
    if (!/.py$/.test(fn.val())) {
        fn.val(fn.val() + ".py");
    }
}
    
var zoom_mode = "geometric";

function change_zoom_mode() {
    if (zoom_mode == "none") {
        zoom_mode = "geometric";
    } else if (zoom_mode == "semantic") {
        zoom_mode = "geometric";
    } else {
        zoom_mode = "semantic";
    }
    $('#zoom_mode').text(zoom_mode);
    
    update_text();
    update_net_sizes();
}

function check_server_for_file_changes() {
    var data = new FormData();
    data.append('filename', $('#filename').val());

    var xhr = new XMLHttpRequest();
    xhr.open('POST', '/modified_time', true);
    xhr.onload = function (event) {
        mtime = parseFloat(this.responseText);
        if (mtime > server_last_modified_time) {
            open_file($('#filename').val());
        }
    };
    xhr.send(data);
}


$(document).ready(function () {
    //initialize file browser
    $('#filebrowser').hide()
    $('#menu_open').click(function () {
        fb = $('#filebrowser');
        fb.toggle(200);
        if (fb.is(":visible")) {
            $('#filebrowser').fileTree({
                root: '.',
                script: '/browse'
            }, open_file);
        }
    })
    $('#menu_save').click(save_file);
    $('#menu_save').addClass('disable');
    $('#filename').change(edit_filename);
    $('#zoom_mode').click(change_zoom_mode);
    $('#zoom_mode').text(zoom_mode);    

    if (gui_server_check_interval>0) {
        window.setInterval(check_server_for_file_changes, 
                           gui_server_check_interval);
    }
});
