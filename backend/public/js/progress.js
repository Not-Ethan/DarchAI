document.querySelectorAll('.delete-task').forEach(function(button) {
    button.addEventListener('click', function(event) {
      var taskId = event.target.getAttribute('data-task-id');
      fetch('/delete-task/' + taskId, {
        method: 'DELETE',
      }).then(function(response) {
        if (response.ok) {
          // Remove the task from the table.
          event.target.closest('tr').remove();
          // Remove the associated view task row as well.
          document.getElementById('collapse' + taskId).remove();
        } else {
          console.error('Failed to delete task', taskId, response);
        }
      });
    });
  });
  