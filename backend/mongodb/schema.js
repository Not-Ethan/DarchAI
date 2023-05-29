const mongoose = require("mongoose");

const userSchema = new mongoose.Schema({
    id: { type: String, required: true, unique: true },
    googleId: { type: String, unique: true },
    username: { type: String, required: true },
    email: { type: String, required: true, unique: true },
    tasks: {type: Array},
    accountType: {type: String, enum: ["free", "premium"], default: "free"},
    subscriptionEnd: {type: Date, default: null},
    monthlyEvidence: {type: Number, default: 0},

  }, { timestamps: true });
const evidenceSchema = new mongoose.Schema({
    id: { type: String, required: true, unique: true },
    tagline: { type: String, required: true },
    url: { type: String, required: true },
    citation: { type: String, required: true },
    relevant_sentences: { type: Array, required: true },
    task: { type: String, required: true},
    used: { type: Boolean, default: false, required: true },
  }, { timestamps: true});
const taskSchema = new mongoose.Schema({
    id: { type: String, required: true, unique: true },
    topic: { type: String },
    side: { type: String },
    argument: { type: String, required: true },
    result: { type: Array },
    status: { type: String },
    user: { type: String, required: true},
  }, { timestamps: true});

const rawEvidenceSchema = new mongoose.Schema({
    id: { type: String, required: true, unique: true },
    fulltext: { type: String, required: true },
    url: { type: String, required: true },
    prompt: { type: String, required: true },
    evidence: {type: String, required: true}
  }, { timestamps: true});

module.exports.User = mongoose.model("user", userSchema);
module.exports.Evidence = mongoose.model("evidence", evidenceSchema);
module.exports.Task = mongoose.model("task", taskSchema);
module.exports.RawEvidence = mongoose.model("rawEvidence", rawEvidenceSchema);