const router = require("express").Router();

router.get("/faq", (req, res) => {
    res.render("faq.ejs", { user: req.user || null, qna: require("./faq.js")})
    delete require.cache[require.resolve("./faq.js")];
});

module.exports = router;