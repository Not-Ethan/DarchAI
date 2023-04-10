document.getElementById('login-form').addEventListener('submit', async (event) => {
    event.preventDefault();
  
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
  
    const response = await fetch('/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    });
  
    const data = await response.json();
    if (data.status === 'success') {
      alert('Login successful!');
      // Redirect to the desired page after successful login
      window.location.href = '/interface';
    } else {
      alert(data.message);
    }
  });
  