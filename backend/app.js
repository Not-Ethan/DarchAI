const express = require("express")
const app = express()
const axios = require('axios')
const uuid = require('uuid');
const zlib = require('zlib');
const base64 = require("base64-js");
const bcrypt = require("bcrypt");
const mongoose = require('mongoose');
const uri = `mongodb://127.0.0.1:${process.env.MONGO_PORT || 27017}/`;
const passport = require('passport');
const GoogleStrategy = require('passport-google-oauth20').Strategy;

const authRoutes = require("./routes/authRoutes.js");
const faqsRoutes = require("./routes/faqRoutes.js");
const { User, Evidence, Task, RawEvidence } = require("./mongodb/schema.js");
const { addTask } = require("./util/task.js");

const PORT = process.env.PORT || 3000
const SERVICE_PORT = process.env.SERVICE_PORT || 5000
const hostname = process.env.HOSTNAME || "localhost"
const service_hostname = process.env.SERVICE_HOSTNAME || "localhost"

async function main() {
await mongoose.connect(uri, {
    useNewUrlParser: true,
    useUnifiedTopology: true
});

Task.deleteMany({status: "processing"}).then(()=>{
  console.log("Deleted all processing tasks");
});

//const bodyParser = require('body-parser');
const {
  Document,
  Packer
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
      callbackURL: `http://${(()=>{if(hostname=="localhost") return hostname + ":" + PORT; else return hostname})()}/auth/google/callback`,
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

const taskQueue = {};

const isLoggedIn = require("./util/isLoggedIn.js");

async function getAiResponse(topic, side, argument, num=10, sentenceModel=0, taglineModel=0) {
    topic = "a";
    side = "sup";

    const apiUrl = `http://${service_hostname}:${SERVICE_PORT}/process`;

    let response;

    try {
      response = await axios.post(apiUrl, { topic, side, argument, num, sentence_model: sentenceModel, tagline_model: taglineModel }, {'content-type': 'application/json'});
    } catch(e) {
      console.log(e);
      return {status: 'error', message: e.response.statusText || 'An unknown error occurred while processing the request'};
    }

    return response.data;
}

async function getTaskStatus(request_id) {

  const apiUrl = `http://${service_hostname}:${SERVICE_PORT}/check_progress`;

  const response = await axios.get(apiUrl+"?task_id="+request_id, {timeout: 10000});

  return response.data;
}

app.use(authRoutes);
app.use(faqsRoutes);

app.get("/", (req, res) => {
    res.render("index.ejs", {user: req.user || null})
})

app.get('/interface', isLoggedIn, (req, res) => {
    res.render('ai.ejs', {user: req.user || null});
});

app.post('/generate-response', isLoggedIn, async (req, res) => {

  const user = await User.findOne({id: req.user.id});

  if(user.accountType == 'free' && taskQueue[req.user.id] && taskQueue[req.user.id].tasks.length >= 1){
    res.status(500).send({message: "Error: You have reached the maximum number of tasks for your account type. Please upgrade your account to continue."});
    return;
  } else if(user.accountType=="paid" && taskQueue[req.user.id] && taskQueue[req.user.id].tasks.length >= 3){
    res.status(500).send({message: "Error: You have reached the maximum number of tasks."});
    return;
  }

  const { topic, side, argument, num } = req.body;

  let config = require('./util/accountConfig.js')(user.accountType);
  if(num > config.MAX_EVIDENCE)
    return res.status(500).send({message: "Error: Your account type does not allow you to request more than "+config.MAX_EVIDENCE+" pieces of evidence."});
  if(num <= 0)
    return res.status(500).send({message: "Error: Invalid number of evidence requested."});
  if(argument.length > config.MAX_ARG_LENGTH)
    return res.status(500).send({message: "Error: Your argument must be less than "+config.MAX_ARG_LENGTH+" characters long."});
  if(argument.length < config.MIN_ARG_LENGTH)
    return res.status(500).send({message: "Error: Your argument must be at least "+config.MIN_ARG_LENGTH+" characters long."});
  
  // Call the Python API and get the response 
  console.log("Calling API...");
  const response = await getAiResponse(topic, side, argument, num);
  console.log("API response: ", response);
  if(response['status'] == 'success'){
    let task_id = response['task_id'];
    // Add the new task to the user's list of tasks
    await addTask(req.user.id, task_id, { topic, side, argument }, taskQueue);
    let reply = {message: "Successfully queued task. Task ID: "+response['task_id'], uuid: response['task_id']}
    res.send(reply);
    console.log("QUEUED TASK: ", taskQueue[req.user.id]);
  } else {
    res.status(500).send({message: "Error: "+response['message']});
  }
});

app.delete("/delete-task/:id", isLoggedIn, async (req, res) => {
  const task_id = req.params.id;
  let task = await Task.findOne({id: task_id});

  if(!task)
    return res.status(500).send({message: "Error: Task not found."});
  if(task.user != req.user.id)
    return res.status(500).send({message: "Error: You do not have permission to delete this task."});
  await task.deleteOne({id: task_id});

  res.status(200).send({message: "Successfully deleted task."});
});

app.get("/logout", (req, res) => {
    req.logout((err)=>{
      if(err) 
        return console.log(err);
      res.redirect("/");
    });
    
})

app.get("/progress", isLoggedIn, async (req, res) => {
  
  let tasks = [];
  if (taskQueue[req.user.id] && taskQueue[req.user.id].tasks) {
    tasks = taskQueue[req.user.id].tasks;
  }

    const tasksPromises = tasks.map((task) => {
      return getTaskStatus(task.id);
    });

    let completedTasks = await Task.find({user: req.user.id, status: "complete"});
    Promise.all(tasksPromises).then((responses) => {
      responses = responses.concat(completedTasks);
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
});

app.get('/get-status/:uuid', async (req, res) => {
  let uuid = req.params.uuid;

  // Call the Python API and get the response
  const response = await getTaskStatus(uuid);
  res.send(response)
});

app.get('/evidence/:userId/:taskId', isLoggedIn, async (req, res) => {
  if(req.params.userId != req.user.id) return res.status(403).send("You are not authorized to view this page");
  // Get the evidence data for the given taskId
  const task = await Task.findOne({user: req.user.id, id: req.params.taskId});

  if(!task) return res.status(404).send("Task not found");
  if(task.result==null) return res.status(404).send("Task not complete");

  let result = [];
  for(let id of task.result) {
    let evidence = await Evidence.findOne({id: id});
    if(evidence) {
      result.push(evidence);
    }
  }

  res.render('evidence.ejs', {user: req.user, taskId: req.params.taskId, data: result, topic: task['topic'] || null, side: task['side'] || null, argument: task['argument'] || null});

});


app.post('/save-evidence', isLoggedIn, async (req, res) => {
  try {
    const data = req.body.data;
    const taskId = req.headers.task;
    let evidenceData = [];

    let task = await Task.findOne({id: taskId})

    for(let id of data) {
      let ev = await Evidence.findOne({id: id});

      if(ev && ev.task==taskId && task && task.user==req.user.id && task.status=="complete" && task.result.includes(id)) {
        evidenceData.push(ev);
      };
    }

    if(evidenceData.length==0) return res.status(404).send("No evidence");
    for(let ev of evidenceData) {
      Evidence.updateOne({id: ev.id}, {used: true});
    }
    const doc = new Document(require("./util/doc.js")(evidenceData, task.argument));


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
  zlib.unzip(buffer, async (err, result) => {

    if (err) {
      console.error("Error decompressing data:", err);
      res.status(500).send("Error decompressing data");
    } else {
      const data = JSON.parse(result.toString());

      const evidenceIds = [];
  // Update the result property of the original task with the array of evidence IDs
  for (const url in data.data) {

    for(const evidenceItem of data.data[url]) {
      const evidenceId = uuid.v4();

      evidenceItem.id = evidenceId;
      evidenceItem.url = url;
      evidenceIds.push(evidenceId);

      let ev = new Evidence({...evidenceItem, task: taskId});
      await ev.save();
      let rawEv = new RawEvidence({fulltext: data.raw_data[url].full_text, prompt: data.raw_data[url].prompt, url: url, evidence: evidenceId, id: uuid.v4()});
      await rawEv.save();
    }

  }

  let userId;
  for(let user in taskQueue) {
    if(!taskQueue[user] || !taskQueue[user].tasks) continue;
    const userTasks = taskQueue[user].tasks;
    console.log("TASKS: ")
    console.log(userTasks)
    const index = userTasks.findIndex(task => task.id === taskId);
    if (index != -1) { 
      taskQueue[user].tasks[index].result = evidenceIds;
      userId = user;
      taskInfo = taskQueue[user].tasks[index];

    try {
      let doc = await Task.findOneAndUpdate({ id: taskId }, { result: evidenceIds, status: 'complete' });
      if (!doc) {
        throw new Error('Task not found in database');
      }
      res.status(200).send('Task completed and stored');
      taskQueue[userId].tasks.splice(index, 1);
      console.log("REMOVED TASK FROM QUEUE")
      console.log(taskQueue[userId])
      console.log("TASK INFO")
      console.log(taskInfo)
      return;
    } catch (err) {
      console.error(err);
      res.status(500).send('Error updating task');
      return;
    }
      
    } else {
      res.status(404).send('Task not found');
      console.log("Task not found: "+taskId)
      console.log(taskQueue[user])
      return;
    }
  }
    }
  });
});




app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server is running at http://0.0.0.0:${PORT}`);
  });

  }

  main().catch(console.error);

  process.on("SIGINT", async () => {
    process.exit(0);
  });
