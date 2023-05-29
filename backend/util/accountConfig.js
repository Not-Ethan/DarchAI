module.exports = (accountType) => {
    let config = {
        MAX_EVIDENCE: 10,
        MAX_TASKS: 30,
        MAX_EVIDENCE_PER_MONTH: 30,
        MAX_ACTIVE_TASKS: 1,
        MAX_ARG_LENGTH: 200,
        MIN_ARG_LENGTH: 10,
    }
    if(accountType=="paid") {
        config.MAX_EVIDENCE = 100;
        config.MAX_TASKS = 100;
        config.MAX_EVIDENCE_PER_MONTH = 120;
        config.MAX_ACTIVE_TASKS = 5;
    }

    return config;
}