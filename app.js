const express = require("express")
const app = express()
const axios = require('axios')
const uuid = require('uuid');

const session = require('express-session');

app.use(
  session({
    secret: 'your-session-secret',
    resave: false,
    saveUninitialized: true,
    cookie: { secure: false },
  })
);
app.use(express.urlencoded({ extended: false }));

const users = {}; // REPLACE THIS WITH DATABASE IN PRODUCTION
const userData = {}; // REPLACE THIS WITH DATABASE IN PRODUCTION

function addTask(userId, taskID, taskData) {
  if (!userData[userId]) {
    userData[userId] = {
      tasks: [],
    };
  }

  const newTask = {
    id: taskID,
    ...taskData
  };

  userData[userId].tasks.push(newTask);
  return newTask;
}

async function getAiResponse(topic, side, argument) {
    // Replace this URL with the actual URL of your Python API
    const apiUrl = 'http://localhost:5000/process';
  
    const response = await axios.post(apiUrl, { topic, side, argument }, {'content-type': 'application/json'});

    return response.data;
}

app.use(express.json())
/*
import { Document, Packer, Paragraph, TextRun } from 'docx';

function createWordDocument(content) {
    const doc = new Document();

    // Apply your formatting here, this is just an example
    const paragraph = new Paragraph();
    const textRun = new TextRun(content);
    paragraph.addRun(textRun);
    doc.addParagraph(paragraph);

    return Packer.toBlob(doc);
}*/


app.use(express.static("public"))
app.set('view engine', 'ejs')

app.get("/", (req, res) => {
    res.render("index.ejs", {user: req.session.user || null})
})
app.get('/interface', isLoggedIn, (req, res) => {
    res.render('interface.ejs', {user: req.session.user || null});
  });


app.post('/generate-response', isLoggedIn, async (req, res) => {
  const { topic, side, argument } = req.body;

  // Call the Python API and get the response
  const response = await getAiResponse(topic, side, argument);
  
  if(response['status'] == 'success'){
    let request_id = response['request_id'];
    // Add the new task to the user's list of tasks
    addTask(req.session.user.id, request_id, { topic, side, argument });

  }

  res.send(getResponseObject(response));
});

async function getTaskStatus(request_id) {

    const apiUrl = 'http://localhost:5000/check_progress';
  
    const response = await axios.get(apiUrl+"?request_id="+request_id);

    return response.data;
}

app.get('/get-status/:uuid', async (req, res) => {
  let uuid = req.params.uuid;

  // Call the Python API and get the response
  const response = await getTaskStatus(uuid);
  console.log(response)
  res.send(getResponseObject(response));
});

function getResponseObject(response) {
  if(response['status'] == 'success'){
    return {message: "Successfully queued task. Task ID: "+response['request_id'], uuid: response['request_id'], status: 0}
  } else if(response['status'] == 'error'){
    return {message: "Error: "+response['message'], status: -1};
  } else if(response['status'] == 'processing'){
    return {message: "Processing; current stage: "+response['progress']['stage'] + "; current progress: "+response['progress']['progress'], status: 1};
  }
}

app.get('/register', (req, res) => {
  res.render('register');
});

app.get('/login', (req, res) => {
  res.render('login');
});


app.post('/register', (req, res) => {
  const { username, password } = req.body;

  if (users[username]) {
    return res.json({ status: 'error', message: 'Username already exists' });
  }

  const userId = uuid.v4();
  users[username] = {
    id: userId,
    password, // Store a password hash in a real application
    name: username
  };
  req.session.user = users[username];
  return res.json({ status: 'success', userId });
});

app.post('/login', (req, res) => {
  const { username, password } = req.body;

  // Replace this with proper user authentication (e.g., verify password hash)
  const user = users[username];
  if (user && user.password === password) {
    req.session.user = user;
    res.json({ status: 'success', userId: user.id });

  } else {
    res.status(401).json({ status: 'error', message: 'Invalid username or password' });
  }
});


function isLoggedIn(req, res, next) {
  if (req.session && req.session.user) {
    next(); // User is logged in, proceed to the next middleware or route
  } else {
    res.redirect("/login")
  }
}


app.listen(3000, '0.0.0.0', () => {
    console.log(`Server is running at http://0.0.0.0:${3000}`);
  });




  /*const wordDocumentBuffer = await createWordDocument(JSON.stringify(aiResponse)); // Adjust this according to your desired format
  res.setHeader('Content-Type', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document');
  res.setHeader('Content-Disposition', 'attachment; filename=ai_response.docx');
  res.send(wordDocumentBuffer);*/