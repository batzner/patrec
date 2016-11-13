// Disable dropzone from autocreating the dropzones
Dropzone.autoDiscover = false;

// Empty the server cache
$.post('/api/empty-cache');

$('.alert').hide();

// Create the images dropzone
$("div#images-dropzone").dropzone({
    acceptedFiles: "image/png, image/jpeg, image/jpg", //This is a comma separated list of mime types or file extensions.Eg.: image/*,application/pdf,.psd.
    dictDefaultMessage: "Drop the images here or click to upload.",
    init: function() {
        var dropzone = this; // Store the dropzone for access in callbacks
        var errorDisplay = $('#form-error');

        this.on("error", function(file, errorMessage) {
            if (file) dropzone.removeFile(file);
            errorDisplay.show().html(errorMessage);
        });
        this.on("success", function (file) {
            errorDisplay.hide();
        });
        this.on('complete', function (file) {
            updateSubmitButton();
        });
    },
    maxFilesize: 100, // MB
    paramName: "file", // The name that will be used to transfer the file
    url: "/api/upload-images"
});

$("div#pattern-dropzone").dropzone({
    acceptedFiles: "image/png, image/jpeg, image/jpg", //This is a comma separated list of mime types or file extensions.Eg.: image/*,application/pdf,.psd.
    dictDefaultMessage: "Drop the pattern here or click to upload.",
    init: function() {
        var dropzone = this; // Store the dropzone for access in callbacks
        var errorDisplay = $('#form-error');

        this.on("error", function(file, errorMessage) {
            if (file) dropzone.removeFile(file);
            errorDisplay.show().html(errorMessage);
        });
        this.on("success", function (file) {
            while (dropzone.files.length > 1) dropzone.removeFile(dropzone.files[0])
            errorDisplay.hide();
        });
        this.on('complete', function (file) {
            updateSubmitButton();
        });
    },
    maxFilesize: 100, // MB
    paramName: "file", // The name that will be used to transfer the file
    url: "/api/upload-pattern"
});

// Get the dropzones
var imagesDropzone = Dropzone.forElement("div#images-dropzone");
var patternDropzone = Dropzone.forElement("div#pattern-dropzone");

updateSubmitButton();

function updateSubmitButton() {
    var button = $('#submit-button');

    if (isReadyForSubmit()) {
        console.log('Enabling Button');
        button
            .removeClass('disabled-button')
            .off()
            .attr('href', '/result')
    } else {
        console.log('Disabling Button');
        button
            .addClass('disabled-button')
            .click(showMissingMessage)
            .attr('href', '#');
    }
}

function isReadyForSubmit() {
    // Check that there is a pattern
    if (!patternDropzone.files.length) return false;

    // Check that there are files
    if (!imagesDropzone.files.length) return false;

    return true;
}

function showMissingMessage() {
    console.log('Showing an error message');
    var errorMessage = $('#form-error');
    if (!patternDropzone.files.length) {
        errorMessage.show().html('Please add a pattern before starting the recognition.');
    } else if (!imagesDropzone.files.length) {
        errorMessage.show().html('Please add some images before starting the recognition.');
    }
}