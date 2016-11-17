$('.alert').hide();

// Display the spinner
var opts = {
  lines: 14 // The number of lines to draw
, length: 15 // The length of each line
, width: 5 // The line thickness
, radius: 30 // The radius of the inner circle
, scale: 1 // Scales overall size of the spinner
, corners: 0 // Corner roundness (0..1)
, color: '#c8cbd7' // #rgb or #rrggbb or array of colors
, opacity: 0.25 // Opacity of the lines
, rotate: 0 // The rotation offset
, direction: 1 // 1: clockwise, -1: counterclockwise
, speed: 1 // Rounds per second
, trail: 60 // Afterglow percentage
, fps: 20 // Frames per second when using setTimeout() as a fallback for CSS
, zIndex: 2e9 // The z-index (defaults to 2000000000)
, className: 'spinner' // The CSS class to assign to the spinner
, top: '50%' // Top position relative to parent
, left: '50%' // Left position relative to parent
, shadow: false // Whether to render a shadow
, hwaccel: false // Whether to use hardware acceleration
, position: 'absolute' // Element positioning
}
var target = document.getElementById('spinner')
var spinner = new Spinner(opts).spin(target);

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
        var resultItem = $('<div class="result-item box"/>');
        resultItem.append($('<div class="img-wrapper"><img src="/api/get-image/'+match.id+'"/></div>'));

        var scoreWrapper = $('<div class="score-wrapper"/>');
        var score = parseInt(match.value);
        scoreWrapper.append($('<div class="score">'+score+'<span class="unit">Votes</span></div>'));
        scoreWrapper.append($('<div class="metric-name">Similarity</div>'));
        resultItem.append(scoreWrapper);

        var matchStatus = '<div class="match-wrapper"><i class="fa fa-check-circle match-icon"></i><div class="metric-name">Match</div></div>';
        if (!match.is_match) {
            matchStatus = '<div class="match-wrapper"><i class="fa fa-times-circle no-match-icon"></i><div class="metric-name">No Match</div></div>';
        }
        resultItem.append($(matchStatus));
        resultList.append(resultItem);
    });
}