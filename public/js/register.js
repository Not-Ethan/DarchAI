document.getElementById('register-form').addEventListener('submit', async (event) => {
    event.preventDefault();
  
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
    const email = document.getElementById('email').value;
  
    const response = await fetch('/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password, email }),
    });
  
    const data = await response.json();
    if (data.status === 'success') {
      alert('Registration successful!');
      window.location.href = '/interface';
    } else {
      alert(data.message);
    }
  });
  