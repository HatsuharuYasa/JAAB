const express = require('express');
const router = express.Router();
const { spawn } = require('child_process');

// WebSocket endpoint for inference logs
router.ws('/run-inference', (ws, req) => {
    const python = spawn('python3', [
        './model/run_inference.py',
        '--model', req.query.model,
        '--input', req.query.input,
        '--output', req.query.output
    ]);

    // Send stdout to client
    python.stdout.on('data', (data) => {
        ws.send(data.toString());
    });

    // Send stderr to client
    python.stderr.on('data', (data) => {
        ws.send(`ERROR: ${data.toString()}`);
    });

    // Notify when done
    python.on('close', (code) => {
        ws.send(`Inference finished with exit code ${code}`);
        ws.close();
    });

    // Handle client disconnect
    ws.on('close', () => {
        python.kill();
    });
});

module.exports = router;
