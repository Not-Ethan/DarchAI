const express = require("express")
const app = express()

app.use(express.static("public"))
app.set('view engine', 'ejs')

app.get("/", (req, res) => {
    res.render(__dirname+"/views/index.html");
})

app.listen(3000, '0.0.0.0', () => {
    console.log(`Server is running at http://0.0.0.0:${3000}`);
  });