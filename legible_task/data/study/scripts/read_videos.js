var fileSystem = require("browserify-fs");
fileSystem.readFile("scripts/videos.json", (err, data) => {
    if(err) {
        console.log("File reading failed", err)
        return
    }
    console.log("File data:", data)
    })