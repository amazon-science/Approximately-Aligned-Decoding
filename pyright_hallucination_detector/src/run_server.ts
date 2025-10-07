import express from "express";

import {HallucinationDetectionRequest, HallucinationResult, PyrightWrapper} from "./check_hallucination.js"

let app = express();
app.use(express.json());

const python_path = process.env.PYTHON_EXECUTABLE || "/home/ec2-user/anaconda3/envs/bigcodebench/bin/python";
const port = parseInt(process.env.PORT || "0", 10);
const wrapper = new PyrightWrapper(python_path, true);

app.post("/detect", (req, res) => {
    let request: HallucinationDetectionRequest = req.body;
    let result: HallucinationResult;

    try {
        result = wrapper.detectHallucinations(request);

    } catch (error) { // Sometimes the file fails to parse at all
        console.log(error);
        result = {
            hallucination: false,
            index_of_hallucination: null,
        }
    }

    res.json(result);
})


const listener = app.listen(port, "127.0.0.1", () => { // @ts-ignore
    console.log("Server started on port " + listener.address().port)
});