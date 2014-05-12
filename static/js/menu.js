//*****************
// Setup the menu
//*****************

//Load the browser and hide it
function open_file(file) {
    $('#filebrowser').hide();

    container.selectAll('.link').remove();
    container.selectAll('.node').remove();
    editor.setValue('');
    $('#filename').val(file)

    var data = new FormData();
    data.append('filename', file);

    var xhr = new XMLHttpRequest();
    xhr.open('POST', '/openfile', true);
    xhr.onload = function (event) {
        editor.setValue(this.responseText);
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
});
