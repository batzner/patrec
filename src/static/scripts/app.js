Dropzone.options.imagesDropzone = {
    paramName: "file", // The name that will be used to transfer the file
    maxFilesize: 100, // MB
    accept: function(file, done) {
        if (file.name == "justinbieber.jpg") {
            done("Naha, you don't.");
        }
        else { done(); }
    },
    dictDefaultMessage: "Drop the images here or click to upload."
};

Dropzone.options.patternDropzone = {
    paramName: "file", // The name that will be used to transfer the file
    maxFilesize: 100, // MB
    accept: function(file, done) {
        if (file.name == "justinbieber.jpg") {
            done("Naha, you don't.");
        }
        else { done(); }
    },
    dictDefaultMessage: "Drop the pattern here or click to upload.",
    maxFiles: 1
};