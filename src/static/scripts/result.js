$('.alert').hide();
$.get('/api/find-matches')
    .done(function(data) {
        console.log(data);
    })
    .fail(function(error) {
        console.log(error)
    });