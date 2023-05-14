const express = require("express")
const app = express()
const axios = require('axios')
const uuid = require('uuid');
const zlib = require('zlib');
const base64 = require("base64-js");
const bcrypt = require("bcrypt");
const {MongoClient} = require('mongodb');
const mongoose = require('mongoose');
const uri = 'mongodb://localhost:27017/';
const passport = require('passport');
const GoogleStrategy = require('passport-google-oauth20').Strategy;

const userSchema = new mongoose.Schema({
  id: { type: String, required: true, unique: true },
  googleId: { type: String, unique: true },
  username: { type: String, required: true },
  email: { type: String, required: true, unique: true },
  password: { type: String },
});

const User = mongoose.model("user", userSchema);

async function main() {
await mongoose.connect(uri, {
    useNewUrlParser: true,
    useUnifiedTopology: true
});
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

const GOOGLE_CLIENT_ID = process.env.GClient_ID;
const GOOGLE_CLIENT_SECRET = process.env.GClient_Secret;

app.use(
  session({
    secret: 'your-session-secret',
    resave: false,
    saveUninitialized: true,
    cookie: { secure: false },
  })
);

app.use(passport.initialize());
app.use(passport.session());

passport.serializeUser((user, done) => {
  done(null, user.id);
});

passport.deserializeUser(async (id, done) => {
  try {
    const user = await User.findOne({id: id});
    done(null, user);
  } catch (err) {
    done(err);
  }
});


passport.use(
  new GoogleStrategy(
    {
      clientID: GOOGLE_CLIENT_ID,
      clientSecret: GOOGLE_CLIENT_SECRET,
      callbackURL: 'http://localhost:3000/auth/google/callback',
    },
    async (accessToken, refreshToken, profile, done) => {
      const { id, displayName, emails } = profile;
      const email = emails[0].value;

      try {
        let user = await User.findOne({ email });

        if (user) {
          // If the user already exists, update the account (if needed) and sign them in.
          done(null, user);
        } else {
          // If the user doesn't exist, create a new account.
          const newUser = new User({ email, username: displayName, googleId: id, id: uuid.v4() });
          await newUser.save();

          done(null, newUser);
        }
      } catch (err) {
        done(err);
      }
    }
  )
);


app.use(express.json({limit: '50mb'}))
app.use(express.urlencoded({ extended: false, limit: '50mb' }));
app.use(express.static("public"))

app.set('view engine', 'ejs')

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
  if (req.user) {
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
    res.render("index.ejs", {user: req.user || null})
})

app.get('/interface', isLoggedIn, (req, res) => {
    res.render('interface.ejs', {user: req.user || null});
});

app.post('/generate-response', isLoggedIn, async (req, res) => {
  const { topic, side, argument, num } = req.body;

  // Call the Python API and get the response
  const response = await getAiResponse(topic, side, argument, num);
  
  if(response['status'] == 'success'){
    let task_id = response['task_id'];
    // Add the new task to the user's list of tasks
    addTask(req.user.id, task_id, { topic, side, argument });
    let reply = {message: "Successfully queued task. Task ID: "+response['task_id'], uuid: response['task_id'], status: 0}
    res.send(reply);
  } else {
    res.status(500).send({message: "Error: "+response['message']});
  }
});

app.get("/logout", (req, res) => {
    req.logout((err)=>{
      if(err) 
        return console.log(err);
      res.redirect("/");
    });
    
})

app.get("/progress", isLoggedIn, (req, res) => {

  let tasks = [];
  if (taskQueue[req.user.id] && taskQueue[req.user.id].tasks) {
    tasks = taskQueue[req.user.id].tasks;
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
      res.render("progress.ejs", { user: req.user || null, tasks: temp });
    });
  } else {
    res.render("progress.ejs", { user: req.user || null, tasks: [] });
  }
});

app.get('/get-status/:uuid', async (req, res) => {
  let uuid = req.params.uuid;

  // Call the Python API and get the response
  const response = await getTaskStatus(uuid);
  res.send(response)
});

app.get('/register', (req, res) => {
  res.render('register', {user: null});
});

app.get('/login', (req, res) => {
  res.render('register', {user: null});
});
app.get('/auth/google', passport.authenticate('google', { scope: ['profile', 'email'] }));

app.get('/auth/google/callback',
passport.authenticate('google', { failureRedirect: '/login' }),
function(req, res) {
  // Successful authentication, redirect to the desired page (e.g., home).
  res.redirect('/');
});

/*
app.post('/register', async (req, res) => {
  if (req.session.user) {
    req.session.destroy();
  }

  const { username, password, email } = req.body;
  let newUser; 

  try {
    const hashedPassword = await bcrypt.hash(password, 10);
    newUser = new User({ email, username, password: hashedPassword, id: uuid.v4() }); // Reassign newUser
    await newUser.save();

    req.session.user = newUser;
    return res.json({ status: 'success', username });

  }  catch(err) {
    if ((err.name === 'MongoError' || err.name === 'MongoServerError') && err.code === 11000) {
      return res.json({ status: 'error', message: 'Email already exists' });
    } else {
      console.error(err);
    }
  }
});
*/

/*
app.post('/login', async (req, res) => {
  const { username, password } = req.body;

  try {
    // Find the user by username
    const user = await User.findOne({ username });

    if (!user) {
      return res.status(404).json({ status: 'error', message: 'User not found' });
    }

    // Check if the provided password is correct
    const isPasswordCorrect = await bcrypt.compare(password, user.password);

    if (!isPasswordCorrect) {
      return res.status(401).json({ status: 'error', message: 'Incorrect password' });
    }

    // Set the user session
    req.session.user = user;

    // Send success response
    res.status(200).json({ status: 'success', userId: user.id });

  } catch (error) {
    console.error(error);
    res.status(500).json({ status: 'error', message: 'An error occurred while logging in' });
  }
});
*/

app.get('/evidence/:userId/:taskId', isLoggedIn, async (req, res) => {
  if(req.params.userId != req.user.id) return res.status(403).send("You are not authorized to view this page");
  // Get the evidence data for the given taskId
  const task = taskQueue[req.params.userId].tasks.find((task) => task.id == req.params.taskId);

  if(!task) return res.status(404).send("Task not found");
  if(task.result==null) return res.status(404).send("Task not complete");
  let result = task.result.map((id) => {
    return evidence[id];
  })

  res.render('evidence.ejs', {user: req.user, taskId: req.params.taskId, data: result, topic: task['result']['topic'], side: task['result']['side'], argument: task['result']['argument'] });

});


app.post('/save-evidence', isLoggedIn, async (req, res) => {
  try {
    const data = req.body.data;
    let evidenceData = [];

    data.forEach((id)=>{
       
      if(evidence[id]&&evidence[id].user == req.user.id) {
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

    if (err) {
      console.error("Error decompressing data:", err);
      res.status(500).send("Error decompressing data");
    } else {
      const data = JSON.parse(result.toString());

      const evidenceIds = [];
  // Update the result property of the original task with the array of evidence IDs
  for (const url in data.data) {

    // Store the raw evidence data for later use in training the AI
    raw_evidence[url] = {full_text: data['raw_data'], sentence_indices: []};


    data.data[url].forEach((evidenceItem, index) => {
      const evidenceId = uuid.v4();

      evidenceItem.id = evidenceId;
      evidenceItem.url = url;
      evidenceIds.push(evidenceId);

      evidence[evidenceId] = evidenceItem;
      console.log("EV Keys: ")
      console.log(Object.keys(evidenceItem))
        for(sentence in evidenceItem['relevant_sentences']) {
          raw_evidence[url]['sentence_indices'].push({start: sentence[4], end: sentence[5]});
        }

      raw_evidence[url]['sentence_indices'].push(index);
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
  console.log(raw_evidence)
});




app.listen(3000, '0.0.0.0', () => {
    console.log(`Server is running at http://0.0.0.0:${3000}`);
  });




  /*const wordDocumentBuffer = await createWordDocument(JSON.stringify(aiResponse)); // Adjust this according to your desired format
  res.setHeader('Content-Type', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document');
  res.setHeader('Content-Disposition', 'attachment; filename=ai_response.docx');
  res.send(wordDocumentBuffer);*/
  }

  main().catch(console.error);

  process.on("SIGINT", async () => {
    process.exit(0);
  });