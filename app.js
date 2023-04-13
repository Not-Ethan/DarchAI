const express = require("express")
const app = express()
const axios = require('axios')
const uuid = require('uuid');
const zlib = require('zlib');
const base64 = require("base64-js");

//const bodyParser = require('body-parser');
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

app.use(
  session({
    secret: 'your-session-secret',
    resave: false,
    saveUninitialized: true,
    cookie: { secure: false },
  })
);
app.use(express.json({limit: '50mb'}))
app.use(express.urlencoded({ extended: false, limit: '50mb' }));
app.use(express.static("public"))

app.set('view engine', 'ejs')

const users = {}; // REPLACE THIS WITH DATABASE IN PRODUCTION
const taskQueue = {}; // REPLACE THIS WITH DATABASE IN PRODUCTION
const evidence = {}; // REPLACE THIS WITH DATABASE IN PRODUCTION
const raw_evidence = {}; // REPLACE THIS WITH DATABASE IN PRODUCTION

function addTask(userId, taskID, taskData) {
  if (!taskQueue[userId]) {
    taskQueue[userId] = {
      tasks: [],
    };
  }

  const newTask = {
    id: taskID,
    ...taskData, // taskData contains topic, side, argument
    result: null
  };

  taskQueue[userId].tasks.push(newTask);
  return newTask;
}

async function getAiResponse(topic, side, argument, num=10) {
    // Replace this URL with the actual URL of your Python API
    const apiUrl = 'http://localhost:5000/process';
  
    const response = await axios.post(apiUrl, { topic, side, argument, num }, {'content-type': 'application/json'});

    return response.data;
}
function isLoggedIn(req, res, next) {
  if (req.session && req.session.user) {
    next(); // User is logged in, proceed to the next middleware or route
  } else {
    res.redirect("/login")
  }
}

async function getTaskStatus(request_id) {

  const apiUrl = 'http://localhost:5000/check_progress';

  const response = await axios.get(apiUrl+"?task_id="+request_id);

  return response.data;
}


app.get("/", (req, res) => {
    res.render("index.ejs", {user: req.session.user || null})
})

app.get('/interface', isLoggedIn, (req, res) => {
    res.render('interface.ejs', {user: req.session.user || null});
});

app.post('/generate-response', isLoggedIn, async (req, res) => {
  const { topic, side, argument, num } = req.body;

  // Call the Python API and get the response
  const response = await getAiResponse(topic, side, argument, num);
  
  if(response['status'] == 'success'){
    let task_id = response['task_id'];
    // Add the new task to the user's list of tasks
    addTask(req.session.user.id, task_id, { topic, side, argument });
    let reply = {message: "Successfully queued task. Task ID: "+response['task_id'], uuid: response['task_id'], status: 0}
    res.send(reply);
  } else {
    res.status(500).send({message: "Error: "+response['message']});
  }
});

app.get("/logout", (req, res) => {
    req.session.destroy();
    res.redirect("/");
})

app.get("/progress", isLoggedIn, (req, res) => {

  let tasks = [];
  if (taskQueue[req.session.user.id] && taskQueue[req.session.user.id].tasks) {
    tasks = taskQueue[req.session.user.id].tasks;
  }

  if (tasks.length > 0) {
    const tasksPromises = tasks.map((task) => {
      if (task.result) {
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
      console.log(responses)
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
            id: response.task_id,
            topic: task.topic,
            side: task.side,
            argument: task.argument,
          };
        } else if (response.status === "error") {
          return { status: "error", message: response.message };
        } else if(response.status==="complete"){
          return response;
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

app.get('/evidence/:userId/:taskId', isLoggedIn, async (req, res) => {
  if(req.params.userId != req.session.user.id) return res.status(403).send("You are not authorized to view this page");
  // Get the evidence data for the given taskId
  const task = taskQueue[req.params.userId].tasks.find((task) => task.id == req.params.taskId);

  if(!task) return res.status(404).send("Task not found");
  if(task.result==null) return res.status(404).send("Task not complete");
  let result = task.result.map((id) => {
    return evidence[id];
  })

  res.render('evidence.ejs', {user: req.session.user, taskId: req.params.taskId, data: result, topic: task['result']['topic'], side: task['result']['side'], argument: task['result']['argument'] });

});


app.post('/save-evidence', isLoggedIn, async (req, res) => {
  try {
    const data = req.body.data;
    let evidenceData = [];

    data.forEach((id)=>{
       
      if(evidence[id]&&evidence[id].user == req.session.user.id) {
        evidenceData.push(evidence[id]);
      };
    })
    // Validate the data, if necessary
    // ...

    // Save the data to the database for later use in training the AI
    // ...
    if(evidenceData.length==0) return res.status(404).send("No evidence");

    const doc = new Document(require("./doc.js")(evidenceData, evidenceData[0]['argument']));


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
  const encodedData = req.body.data;
  const compressedData = base64.toByteArray(encodedData);
  // Decompress the data
  const buffer = Buffer.from(compressedData, 'base64');
  zlib.unzip(buffer, (err, result) => {
    console.log(result)
    if (err) {
      console.error("Error decompressing data:", err);
      res.status(500).send("Error decompressing data");
    } else {
      const data = JSON.parse(result.toString());

      const evidenceIds = [];
  // Update the result property of the original task with the array of evidence IDs
  for (const url in data.data) {

    // Store the raw evidence data for later use in training the AI
    raw_evidence[url] = data['raw_data'];

    data.data[url].forEach((evidenceItem, index) => {
      const evidenceId = uuid.v4();

      evidenceItem.id = evidenceId;
      evidenceItem.url = url;
      evidenceIds.push(evidenceId);

      evidence[evidenceId] = evidenceItem;
    });
  }

  let userId;
  let taskInfo;
  for(let user in taskQueue) {
    if(!taskQueue[user] || !taskQueue[user].tasks) continue;
    const userTasks = taskQueue[user].tasks;
    const index = userTasks.findIndex(task => task.id === taskId);
    if (index != -1) { 
      taskQueue[user].tasks[index].result = evidenceIds;
      userId = user;
      taskInfo = taskQueue[user].tasks[index];
      res.status(200).send('Task completed and stored');
    } else {
      res.status(404).send('Task not found');
    }
  }
  for(let evId of evidenceIds) {
    evidence[evId]['user'] = userId;
    evidence[evId]['topic'] = taskInfo['topic'];
    evidence[evId]['side'] = taskInfo['side'];
    evidence[evId]['argument'] = taskInfo['argument'];
  }
    }
  });
});




app.listen(3000, '0.0.0.0', () => {
    console.log(`Server is running at http://0.0.0.0:${3000}`);
  });




  /*const wordDocumentBuffer = await createWordDocument(JSON.stringify(aiResponse)); // Adjust this according to your desired format
  res.setHeader('Content-Type', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document');
  res.setHeader('Content-Disposition', 'attachment; filename=ai_response.docx');
  res.send(wordDocumentBuffer);*/