// Empty the server cache
$.post('/api/empty-cache');

$('.alert').hide();

currentPattern = null;

Dropzone.options.imagesDropzone = {
    accept: function(file, done) {
        if (file.name == "justinbieber.jpg") {
            done("Naha, you don't.");
        }
        else { done(); }
    },
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
    },
    maxFilesize: 100, // MB
    paramName: "file" // The name that will be used to transfer the file
};

Dropzone.options.patternDropzone = {
    accept: function(file, done) {
        if (file.name == "justinbieber.jpg") {
            done("Naha, you don't.");
        }
        else { done(); }
    },
    acceptedFiles: "image/png, image/jpeg, image/jpg", //This is a comma separated list of mime types or file extensions.Eg.: image/*,application/pdf,.psd.
    dictDefaultMessage: "Drop the pattern here or click to upload.",
    init: function() {
        var dropzone = this; // Store the dropzone for access in callbacks
        var errorDisplay = $('#form-error');
        this.on("error", function(file, errorMessage) {
            if (file) {
                dropzone.removeFile(file);
                currentPattern = null;
            }
            errorDisplay.show().html(errorMessage);
        });
        this.on("success", function (file) {
            if (currentPattern) dropzone.removeFile(currentPattern);
            currentPattern = file;
            errorDisplay.hide();
        });
    },
    maxFilesize: 100, // MB
    paramName: "file" // The name that will be used to transfer the file
};