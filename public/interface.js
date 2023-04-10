$(document).ready(function () {
    $('#submit_new_task').on('submit', async function (event) {
      event.preventDefault();
  
      const topic = $('#topic').val();
      const side = $('#side').val();
      const argument = $('#argument').val();
      console.log("topic: " + topic + " side: " + side + " argument: " + argument)
      // Call your AI function here, and get the response
    // Call your AI function here, and get the response
    try {
      const response = await axios.post('/generate-response', {
        topic,
        side,
        argument,
      });

      // Display the response in the output section
      $('#ai-response').text(response.data.message);

      console.log(response)
      if(response.data.status==0) {
        $("#check-status").on("click", async ()=>{
          const res = await axios.get("/get-status/"+response.data.uuid);
          $('#ai-response').text(res.data.message);
        });
      }
      $('#output-section').show();
      
      
    } catch (error) {
      console.error('Error while sending request:', error);
    }
    });
  });
  