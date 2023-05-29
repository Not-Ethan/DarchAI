const express = require("express");
const router = express.Router();
const { isLoggedIn } = require("../middlewares/auth");
router.get("/progress", isLoggedIn, (req, res) => {
  
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

module.exports = router;