$('.alert').hide();

// Load the matches
$.get('/api/find-matches')
    .done(function(data) {
        console.log(data);
        fillMatches(data);
    })
    .fail(function(error) {
        console.log(error)
    });

function fillMatches(data) {
    var resultList = $('#result-list');
    resultList.empty();
    data.forEach(function (match) {
        var resultItem = $('<li class="result-item"/>');
        resultItem.append($('<img src="/api/get-image/'+match.id+'"/>'));
        var score = parseInt(match.value*100);
        var isMatch = match.is_match ? 'yes' : 'no';
        resultItem.append($('<span>Score: '+score+'% Match: '+isMatch+'</span>'));
        resultList.append(resultItem);
    });
}