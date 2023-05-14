//IMPORTS
const express = require("express")
const app = express()
const axios = require('axios')
const uuid = require('uuid');
const {
  Document,
  Packer,
  Paragraph,
  TextRun,
  UnderlineType,
  BorderStyle, 
  WidthType
} = require('docx');
const session = require('express-session');
const docFormatter = require('./doc.js');


//MIDDLEWARE
app.use(
    session({
      secret: 'your-session-secret',
      resave: false,
      saveUninitialized: true,
      cookie: { secure: false },
    })
  );
app.use(express.urlencoded({ extended: false }));
app.use(express.json())
app.use(express.static("public"))
app.set('view engine', 'ejs')

const users = {}; // REPLACE THIS WITH DATABASE IN PRODUCTION
const userData = {}; // REPLACE THIS WITH DATABASE IN PRODUCTION
let completedTasks = {}; // REPLACE THIS WITH DATABASE IN PRODUCTION
const evidence = {}; // REPLACE THIS WITH DATABASE IN PRODUCTION

app.get("/", (req, res) => {
    res.render("index.ejs", {user: req.session.user || null})
})
app.get('/interface', isLoggedIn, (req, res) => {
    res.render('interface.ejs', {user: req.session.user || null});
  });

//helper functions
async function getAiResponse(topic, side, argument, num=10) {
    // Replace this URL with the actual URL of your Python API
    const apiUrl = 'http://localhost:5000/process';
  
    const response = await axios.post(apiUrl, { topic, side, argument, num }, {'content-type': 'application/json'});

    return response.data;
}

function getResponseObject(response) {
    if(response['status'] == 'success'){
      return {message: "Successfully queued task. Task ID: "+response['request_id'], uuid: response['request_id'], status: 0}
    } else if(response['status'] == 'error'){
      return {message: "Error: "+response['message'], status: -1};
    } else if(response['status'] == 'processing'){
      return {message: "Processing; current stage: "+response['progress']['stage'] + "; current progress: "+response['progress']['progress'], status: 1};
    }
  }



  //GET ENDPOINTS
app.get('/register', (req, res) => {
res.render('register');
});

app.get('/login', (req, res) => {
res.render('login');
});
app.get("/logout", (req, res) => {
    req.session.destroy();
    res.redirect("/");
})

//REWRITE
app.get("/progress", isLoggedIn, (req, res) => {
    let tasks = [];
    if (userData[req.session.user.id] && userData[req.session.user.id].tasks) {
      tasks = userData[req.session.user.id].tasks;
    }
  
    if (tasks.length > 0) {
      const tasksPromises = tasks.map((task) => {
        if (completedTasks[task.id]) {
          return Promise.resolve({
            status: "complete",
            id: task.id,
            topic: task.topic,
            side: task.side,
            argument: task.argument,
          });
        }
        return getTaskStatus(task.id);
      });
  
      Promise.all(tasksPromises).then((responses) => {
        const temp = responses.map((response, index) => {
          const task = tasks[index];
          if (response.status === "processing") {
            return {
              status: "processing",
              progress: {
                stage: response.progress.stage,
                progress_as_string: response.progress.progress,
                progress_as_num: response.progress.as_num,
                progress_raw: {
                  current: response.progress.num,
                  total: response.progress.outof,
                },
              },
              id: response.id,
              topic: task.topic,
              side: task.side,
              argument: task.argument,
            };
          } else if (response.status === "error") {
            return { status: "error", message: response.message };
          } else if (response.status === "complete") {
            return {
              status: "complete",
              id: response.id,
              topic: task.topic,
              side: task.side,
              argument: task.argument,
            };
          }
        });
  
        res.render("progress.ejs", { user: req.session.user || null, tasks: temp });
      });
    } else {
      res.render("progress.ejs", { user: req.session.user || null, tasks: [] });
    }
  });

  app.get('/get-status/:uuid', async (req, res) => {
    let uuid = req.params.uuid;
  
    // Call the Python API and get the response
    const response = await getTaskStatus(uuid);
    res.send(response)
  });

  app.get('/evidence/:userId/:taskId', isLoggedIn, async (req, res) => {
    if(req.params.userId != req.session.user.id) return res.status(403).send("You are not authorized to view this page");
    // Get the evidence data for the given taskId
    const task = userData[req.params.userId].tasks.filter((task) => task.id == req.params.taskId);
  
    let currentTask = await getTaskStatus(req.params.taskId);
  
    if(!task) return res.status(404).send("Task not found");
    if(currentTask['status'] != 'complete') 
      return res.status(404).send("Task not complete");
  
    let result = currentTask['result']['data'];
  
  
    res.render('evidence.ejs', {user: req.session.user, taskId: req.params.taskId, data: result, topic: currentTask['result']['topic'], side: currentTask['result']['side'], argument: currentTask['result']['argument'] });
  
  });



  //POST ENDPOINTS
  app.post('/generate-response', isLoggedIn, async (req, res) => {
    const { topic, side, argument, num } = req.body;
  
    // Call the Python API and get the response
    const response = await getAiResponse(topic, side, argument, num);
    
    if(response['status'] == 'success'){
      let request_id = response['request_id'];
      // Add the new task to the user's list of tasks
      addTask(req.session.user.id, request_id, { topic, side, argument });
  
    }
  
    res.send(getResponseObject(response));
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

  app.post('/save-evidence', isLoggedIn, async (req, res) => {
    try {
      const evidence = req.body.data;
      evidence.forEach((id) => {
        if(!userData[req.session.user.id].evidence.includes(id))
        return
      })
      // Validate the data, if necessary
      // ...
  
      // Save the data to the database for later use in training the AI
      // ...
  
      const doc = new Document(require("./doc.js")(evidenceData));
                                
  
      // Generate the Word document
      const buffer = await Packer.toBuffer(doc);
  
      // Set the response headers
      res.setHeader('Content-Disposition', 'attachment; filename=evidence.docx');
      res.setHeader('Content-Type', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document');
  
      // Send the Word document as a response
      res.send(buffer);
    } catch (error) {
      console.error('Error generating Word document:', error);
      res.status(500).send('Error generating Word document');
    }
  });

  app.post('/task-completed', async (req, res) => {
    const taskId = req.body.taskId;
    const data = req.body.data;
  
    // Make sure the task data is in the correct format
    for (const url in data.data) {
      data[url].forEach((evidenceItem, index) => {
        const evidenceId = uuid.v4();
        evidence[evidenceId] = evidenceItem;
        evidenceItem.id = evidenceId;
        evidenceItem.url = url;
      });
    }
  
    // Save the formatted task data
    completedTasks[taskId] = {
      result: {
        data: data.data,
        topic: req.body.topic,
        side: req.body.side,
        argument: req.body.argument
      }
    };
    res.status(200).send('Task completed and stored');
  });

  app.listen(3000, '0.0.0.0', () => {
    console.log(`Server is running at http://0.0.0.0:${3000}`);
  });