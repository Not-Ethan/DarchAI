module.exports = function isLoggedIn(req, res, next) {
    if (req.user) {
      next(); // User is logged in, proceed to the next middleware or route
    } else {
      res.redirect("/login")
    }
  }