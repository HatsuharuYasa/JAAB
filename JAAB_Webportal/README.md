# JAAB Portal

A modern web-based video processing portal that allows users to upload videos, run AI inference on the JAAB, and manage their video library with an intuitive interface.

## Features

### Video Management
- **Drag & Drop Upload**: Easy video file uploading with progress tracking
- **Smart File Naming**: Automatic duplicate handling with "copy" naming convention
- **Thumbnail Generation**: Server-side thumbnail creation and caching
- **Video Library**: Organized display of uploaded videos and inference results
- **Context Menu**: Right-click to delete videos with confirmation

### Search & Discovery
- **Real-time Search**: Search through video filenames with instant results
- **Smart Filtering**: Shows related inference results when searching uploads
- **Responsive Grid**: Adaptive video card layout for different screen sizes

### Video Player
- **Modal Player**: Popup video player with custom controls
- **Fullscreen Support**: Native fullscreen mode with ESC key support
- **Progress Tracking**: Visual progress bar and time display
- **Keyboard Controls**: ESC to close, space to play/pause

### AI Inference
- **One-Click Processing**: Run AI inference on uploaded videos
- **Real-time Progress**: Live WebSocket updates during processing
- **Automatic Encoding**: Post-processing with FFmpeg for web compatibility
- **Result Management**: Organized storage of inference results

### User Experience
- **Dark Theme**: Modern dark interface with blue accent colors
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Tooltips**: Hover to see full filenames when truncated
- **Progress Indicators**: Visual feedback for uploads and processing
- **Error Handling**: Graceful error messages and recovery

## Prerequisites

- **Node.js** (v14 or higher)
- **Python 3.8+** with conda/miniconda
- **FFmpeg** (for video processing and thumbnail generation)
- **AI Model**: JAAB environment with Focus_small model

## Installation

### 1. Clone the Repository

### 2. Install Backend Dependencies
\`\`\`bash
cd backend
npm install
\`\`\`

### 3. Set Up Python Environment
\`\`\`bash
# Create conda environment (if not exists)
conda create -n JAAB python=3.8
conda activate JAAB

# Install your AI model dependencies here
# (This depends on your specific JAAB setup)
\`\`\`

### 4. Install FFmpeg
**Ubuntu/Debian:**
\`\`\`bash
sudo apt update
sudo apt install ffmpeg
\`\`\`

**macOS:**
\`\`\`bash
brew install ffmpeg
\`\`\`

**Windows:**
Download from [FFmpeg official website](https://ffmpeg.org/download.html)

### 5. Configure Paths
Update the Python path in `backend/server.js` line 203:
\`\`\`javascript
const command = `/path/to/your/conda/envs/TDEED/bin/python model/run_inference.py --model Focus_small --input "${inputPath}" --output "${outputPath}"`
\`\`\`

## Project Structure

\`\`\`
jaab-portal/
├── backend/
│   ├── server.js              # Express server with WebSocket support
│   ├── package.json           # Node.js dependencies
│   ├── uploads/               # Uploaded video files
│   ├── results/               # AI inference results
│   ├── thumbnails/            # Generated thumbnails
│   │   ├── uploads/           # Thumbnails for uploaded videos
│   │   └── results/           # Thumbnails for result videos
│   └── model/                 # AI model files
│       └── run_inference.py   # Python inference script
├── frontend/
│   ├── index.html             # Main HTML file
│   ├── script.js              # Frontend JavaScript
│   ├── main.css               # Styling
│   └── placeholder.svg        # Placeholder image for videos
├── README.md                  # This file
└── requirements.txt           # Python dependencies
\`\`\`

## Usage

### 1. Start the Server
\`\`\`bash
cd backend
npm start
\`\`\`

The server will start on `http://localhost:3000`

### 2. Upload Videos
- Drag and drop video files onto the upload zone
- Or click the upload zone to select files
- Supported formats: MP4, AVI, MOV, etc.

### 3. Run Inference
- Click on any uploaded video to open the player
- Click the "Run Inference" button to process the video
- Monitor progress in the real-time log window

### 4. Manage Videos
- Search videos using the search bar
- Right-click videos to delete them
- View video metadata (duration, date added, type)

## API Endpoints

### Video Management
- `POST /upload` - Upload a video file
- `GET /videos` - Get list of all videos with metadata
- `DELETE /delete-video?filename=<name>&type=<uploads|results>` - Delete a video

### Thumbnails
- `GET /generate-thumbnail?filename=<name>&type=<uploads|results>` - Generate thumbnail
- `GET /thumbnails/<type>/<filename>` - Serve thumbnail images

### Inference
- `WS /inference/run` - WebSocket endpoint for real-time inference progress

## Configuration

### Debug Mode
Toggle debug mode in `frontend/script.js`:
\`\`\`javascript
const DEBUG_MODE = false  // Set to true for detailed logs
\`\`\`

### File Upload Limits
Modify multer configuration in `backend/server.js` for file size limits.

### Video Processing
Customize FFmpeg parameters in the WebSocket inference handler for different quality/speed trade-offs.

## Development

### Frontend Development
The frontend uses vanilla JavaScript with:
- Modern ES6+ features
- WebSocket for real-time communication
- CSS Grid for responsive layouts
- Font Awesome for icons

### Backend Development
The backend uses:
- Express.js for HTTP server
- express-ws for WebSocket support
- Multer for file uploads
- Child processes for Python/FFmpeg integration

### Adding New Features
1. Update the frontend UI in `frontend/index.html` and `frontend/main.css`
2. Add JavaScript functionality in `frontend/script.js`
3. Create new API endpoints in `backend/server.js`
4. Test thoroughly with different video formats and sizes

## Troubleshooting

### Common Issues

**Videos won't play:**
- Ensure FFmpeg is installed and accessible
- Check video format compatibility
- Verify file permissions in uploads/results directories

**Inference fails:**
- Confirm Python environment path is correct
- Ensure T-DEED2 environment is properly set up
- Check model files are in the correct location

**Thumbnails not generating:**
- Verify FFmpeg installation
- Check write permissions in thumbnails directory
- Ensure video files are not corrupted

**Upload fails:**
- Check available disk space
- Verify file size limits
- Ensure uploads directory exists and is writable

### Logs and Debugging
- Enable debug mode for detailed console logs
- Check browser developer tools for frontend errors
- Monitor server console for backend errors
- Use WebSocket messages for inference debugging

## Performance Optimization

### For Large Video Files
- Increase Node.js memory limit: `node --max-old-space-size=4096 server.js`
- Configure nginx for large file uploads in production
- Consider implementing chunked uploads for very large files

### For Many Videos
- Implement pagination for video lists
- Add database for metadata storage
- Consider CDN for thumbnail serving

## License

MIT License

Copyright (c) 2024 Jaime Mejia, Sydney Bailey, Rasarles Nisbett, Gede Wirayasa/ National Dong Hwa University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Support

For issues and questions:
- email james.mejia.73@gmail.com

---

**JAAB Portal** 
