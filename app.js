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
    ...taskData,
    result: null
  };

  userData[userId].tasks.push(newTask);
  return newTask;
}

async function getAiResponse(topic, side, argument, num=10) {
    // Replace this URL with the actual URL of your Python API
    const apiUrl = 'http://localhost:5000/process';
  
    const response = await axios.post(apiUrl, { topic, side, argument, num }, {'content-type': 'application/json'});

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

app.get("/logout", (req, res) => {
    req.session.destroy();
    res.redirect("/");
})

async function getTaskStatus(request_id) {

    const apiUrl = 'http://localhost:5000/check_progress';
  
    const response = await axios.get(apiUrl+"?request_id="+request_id);

    return response.data;
}
app.get("/progress", isLoggedIn, (req, res)=>{
    let tasks = []
    if(userData[req.session.user.id] && userData[req.session.user.id].tasks){
      let temp = userData[req.session.user.id].tasks;
      for(let task of temp){
        tasks.push(getTaskStatus(task.id))
      }
    }
    if(tasks.length > 0){
      Promise.all(tasks).then((responses) => {
        let temp = []
        responses.forEach((response) => {
          if(response['status']=="processing") {
            temp.push({"status": "processing",
             "progress": {
              "stage": response['progress']['stage'],
              "progress_as_string": response['progress']['progress'],
               "progress_as_num": response['progress']['as_num'],
                "progress_raw": {
                  "current": response['progress']['num'],
                 "total": response['progress']['outof']
                }
              },
              "id": response['id'],
              "topic": userData[req.session.user.id].tasks.filter((task) => task.id == response['id'])[0]['topic'],
              "side": userData[req.session.user.id].tasks.filter((task) => task.id == response['id'])[0]['side'],
              "argument": userData[req.session.user.id].tasks.filter((task) => task.id == response['id'])[0]['argument']
            })
          } else if(response['status']=="error") {
             temp.push({"status": "error", "message": response['message']})
          } else if(response['status']=="complete") {
            temp.push({"status": "complete", "id": response['id'], "topic": response['result']['topic'], "side": response['result']['side'], "argument": response['result']['argument']})
          }
        })

        res.render("progress.ejs", {user: req.session.user || null, tasks: temp})
      })
    } else {
      res.render("progress.ejs", {user: req.session.user || null, tasks: []})
    }
})
app.get('/get-status/:uuid', async (req, res) => {
  let uuid = req.params.uuid;

  // Call the Python API and get the response
  const response = await getTaskStatus(uuid);
  res.send(response)
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

app.get('/evidence/:userId/:taskId', isLoggedIn, async (req, res) => {
  // Get the evidence data for the given taskId
  const task = userData[req.params.userId].tasks.filter((task) => task.id == req.params.taskId)[0];

  let currentTask = await getTaskStatus(req.params.taskId);

  if(!task) return res.status(404).send("Task not found");
  if(currentTask['status'] != 'complete') 
    return res.status(404).send("Task not complete");

  let result = currentTask['result']['data'];


  res.render('evidence.ejs', {user: req.session.user, taskId: req.params.taskId, data: result, topic: currentTask['result']['topic'], side: currentTask['result']['side'], argument: currentTask['result']['argument'] });

});


app.post('/save-evidence', isLoggedIn, async (req, res) => {
  try {
    const evidenceData = req.body.data;

    // Validate the data, if necessary
    // ...

    // Save the data to the database for later use in training the AI
    // ...

    const doc = new Document({
      sections: evidenceData.map((evidence) => ({
        properties: {},
        children: [
          new Paragraph({
            children: [
              new TextRun({
                text: evidence.data.tagline,
                bold: true,
                size: 30,
                color: '2460bf',
              }),
            ],
            border: {
              bottom: {
                color: 'auto',
                space: 1,
                value: 'single',
                size: 6,
                style: BorderStyle.SINGLE,
              },
            },
            spacing: {
              after: 400,
            },
          }),
          new Paragraph({
            children: [
              new TextRun({
                text: evidence.url,
                underline: {
                  type: 'single',
                },
                size: 26,
                color: '24a3c9',
              }),
              new TextRun('\n'),
            ],
            spacing: {
              after: 400,
            },
          }),
          new Paragraph({
            children: evidence.data.relevant_sentences.flatMap(([sentence, isRelevant, previousContext, afterContext], sentenceIndex, allSentences) => {
              const children = [];
    
              previousContext.forEach(([context, similarity]) => {
                if (similarity > 0.5) {
                  children.push(
                    new TextRun({
                      text: context + ' ',
                      underline: {
                        type: 'single',
                      },
                      size: 24,
                    })
                  );
                } else {
                  children.push(
                    new TextRun({
                      text: context + ' ',
                      size: 18,
                    })
                  );
                }
              });
    
              if (isRelevant) {
                children.push(
                  new TextRun({
                    text: sentence,
                    bold: true,
                    size: 24,
                  })
                );
              } else {
                children.push(
                  new TextRun({
                    text: sentence,
                    size: 18,
                  })
                );
              }
    
              afterContext.forEach(([context, similarity]) => {
                if (similarity > 0.5) {
                  children.push(
                    new TextRun({
                      text: context + ' ',
                      underline: {
                        type: 'single',
                      },
                      size: 24,
                    })
                  );
                } else {
                  children.push(
                    new TextRun({
                      text: context + ' ',
                      size: 12,
                    })
                  );
                }
              });
    
              if (sentenceIndex < allSentences.length - 1) {
                children.push(new TextRun('[...] '));
              }
    
              return children;
            }),
            spacing: {
              line: 320, // 1.5 line spacing (in twips)
            },
          }),
        ],
      })),
    });
                              

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
  completedTasks[taskId] = {};

  for (const url in data) {
    data[url].forEach((evidenceItem, index) => {
      const evidenceId = uuid.v4();
      tempStorage[evidenceId] = evidenceItem;
      evidenceItem.id = evidenceId;
    });
  }

  completedTasks[taskId].evidenceData = data;
  res.status(200).send('Task completed and stored');
});


app.listen(3000, '0.0.0.0', () => {
    console.log(`Server is running at http://0.0.0.0:${3000}`);
  });




  /*const wordDocumentBuffer = await createWordDocument(JSON.stringify(aiResponse)); // Adjust this according to your desired format
  res.setHeader('Content-Type', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document');
  res.setHeader('Content-Disposition', 'attachment; filename=ai_response.docx');
  res.send(wordDocumentBuffer);*/