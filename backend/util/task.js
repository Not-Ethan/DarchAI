const { Task } = require("../mongodb/schema");
const axios = require("axios");

module.exports.addTask = async function addTask(userId, taskID, taskData, taskQueue) {
    let t = new Task({id: taskID, topic: taskData.topic, side: taskData.side, argument: taskData.argument, result: null, status: "processing", user: userId});
    await t.save();
  
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

  module.exports.getAiResponse = async function getAiResponse(topic, side, argument, num=10, sentenceModel=0, taglineModel=0) {
    if(!topic)
      topic = "a";
    if(!side)
      side = "sup";
  
      const apiUrl = 'http://localhost:5000/process';
  
      let response;
  
      try {
        response = await axios.post(apiUrl, { topic, side, argument, num, sentence_model: sentenceModel, tagline_model: taglineModel }, {'content-type': 'application/json'});
      } catch(e) {
        return {status: 'error', message: e.response.statusText || 'An unknown error occurred while processing the request'};
      }
  
      return response.data;
  }