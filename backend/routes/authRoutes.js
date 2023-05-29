const express = require('express');
const router = express.Router();
const passport = require('passport');

router.get('/login', (req, res) => {
    if(req.user) return res.redirect('/');
    res.render('login', {user: null});
  });

  router.get('/auth/google', passport.authenticate('google', { scope: ['profile', 'email'] }));

router.get('/auth/google/callback',
passport.authenticate('google', { failureRedirect: '/login' }),
function(req, res) {
  // Successful authentication, redirect to the desired page (e.g., home).
  res.redirect('/');
});

module.exports = router;