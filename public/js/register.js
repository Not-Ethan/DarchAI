function onSignIn(googleUser) {
  // Get the user's ID token, which you can send to your server
  const id_token = googleUser.getAuthResponse().id_token;
  console.log("ID Token: " + id_token);
}
