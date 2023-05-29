$(document).ready(function() {
    $('#aiForm').on('submit', function(e) {
      e.preventDefault();
  
      const argument = $('#argument').val();
      const num = $('#num').val();
      const sentenceModel = $('#sentenceModel').val();
      const taglineModel = $('#taglineModel').val();
  
      $.ajax({
        url: '/generate-response',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ argument, num, sentenceModel, taglineModel }),
        success: function(response) {
          $('#result').html('<div class="alert alert-success alert-dismissible fade show" role="alert">' + 
          `${response.message}` + 
          '<button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>' +
          '</div>');
        },
        error: function(response) {
          $('#result').html('<div class="alert alert-danger alert-dismissible fade show" role="alert">' + 
          `${response.responseJSON.message}` + 
          '<button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>' +
          '</div>');
        }
      });
    });
  });
  