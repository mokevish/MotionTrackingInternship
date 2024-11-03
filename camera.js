import * as posedetection from '@tensorflow-models/pose-detection';
import * as params from './params';

export class Context {
  constructor() {
    this.video = document.getElementById('video');
    this.canvas = document.getElementById('output');
    this.source = document.getElementById('currentVID');
    this.ctx = this.canvas.getContext('2d');
    this.keypointsData = [];
    this.isRecording = false;
    this.startTime = null; // Store the start time in seconds

    const stream = this.canvas.captureStream();
    const options = { mimeType: 'video/webm; codecs=vp9' };
    this.mediaRecorder = new MediaRecorder(stream, options);

    this.mediaRecorder.ondataavailable = this.handleDataAvailable.bind(this);
    this.mediaRecorder.onstop = this.handleStop.bind(this);
  }

  drawCtx() {
    this.ctx.drawImage(
      this.video, 0, 0, this.video.videoWidth, this.video.videoHeight
    );
  }

  clearCtx() {
    this.ctx.clearRect(0, 0, this.video.videoWidth, this.video.videoHeight);
  }

  drawResults(poses) {
    for (const pose of poses) {
      this.drawResult(pose);
    }
  }

  drawResult(pose) {
    if (pose.keypoints != null) {
      this.drawKeypoints(pose.keypoints);
      this.drawSkeleton(pose.keypoints);

      // Store keypoints data
      this.storeKeypoints(pose.keypoints);
    }
  }

  storeKeypoints(keypoints) {
    if (!keypoints.length) return;

    // Calculate elapsed time since recording started in seconds
    const currentTime = (performance.now() - this.startTime) / 1000; 
    const formattedTime = currentTime.toFixed(3); // Keep 3 decimal places for seconds

    const keypointsRow = [formattedTime]; // Start with the timestamp

    keypoints.forEach(kp => {
      if (kp && kp.score != null && kp.score >= (params.STATE.modelConfig.scoreThreshold || 0)) {
        keypointsRow.push(kp.x, kp.y, kp.score); // Add coordinates and score
      } else {
        keypointsRow.push('', '', ''); // If keypoint is invalid or score is low, add empty values
      }
    });

    this.keypointsData.push(keypointsRow.join(','));
  }

  drawKeypoints(keypoints) {
    const keypointInd =
      posedetection.util.getKeypointIndexBySide(params.STATE.model);
    this.ctx.fillStyle = 'White';
    this.ctx.strokeStyle = 'White';
    this.ctx.lineWidth = params.DEFAULT_LINE_WIDTH;

    for (const i of keypointInd.middle) {
      this.drawKeypoint(keypoints[i]);
    }

    this.ctx.fillStyle = 'Green';
    for (const i of keypointInd.left) {
      this.drawKeypoint(keypoints[i]);
    }

    this.ctx.fillStyle = 'Orange';
    for (const i of keypointInd.right) {
      this.drawKeypoint(keypoints[i]);
    }
  }

  drawKeypoint(keypoint) {
    const score = keypoint.score != null ? keypoint.score : 1;
    const scoreThreshold = params.STATE.modelConfig.scoreThreshold || 0;

    if (keypoint && score >= scoreThreshold) {
      const circle = new Path2D();
      circle.arc(keypoint.x, keypoint.y, params.DEFAULT_RADIUS, 0, 2 * Math.PI);
      this.ctx.fill(circle);
      this.ctx.stroke(circle);
    }
  }

  drawSkeleton(keypoints) {
    this.ctx.fillStyle = 'White';
    this.ctx.strokeStyle = 'White';
    this.ctx.lineWidth = params.DEFAULT_LINE_WIDTH;

    posedetection.util.getAdjacentPairs(params.STATE.model).forEach(([i, j]) => {
      const kp1 = keypoints[i];
      const kp2 = keypoints[j];

      const score1 = kp1 && kp1.score != null ? kp1.score : 1;
      const score2 = kp2 && kp2.score != null ? kp2.score : 1;
      const scoreThreshold = params.STATE.modelConfig.scoreThreshold || 0;

      if (kp1 && kp2 && score1 >= scoreThreshold && score2 >= scoreThreshold) {
        this.ctx.beginPath();
        this.ctx.moveTo(kp1.x, kp1.y);
        this.ctx.lineTo(kp2.x, kp2.y);
        this.ctx.stroke();
      }
    });
  }

  start() {
    this.keypointsData = [];
    this.isRecording = true;
    this.startTime = performance.now(); // Use high-resolution time to start the timestamp
    this.mediaRecorder.start();
  }

  stop() {
    this.isRecording = false;
    this.mediaRecorder.stop();
  }

  handleDataAvailable(event) {
    console.log('Data available:', event.data.size);
  }

  handleStop() {
    setTimeout(() => this.exportCSV(), 1000);
  }

  exportCSV() {
    if (this.keypointsData.length === 0) {
      console.log('No keypoints data to export.');
      return;
    }

    const csvHeader = 'timestamp,nose_x,nose_y,nose_score,left_eye_x,left_eye_y,left_eye_score,right_eye_x,right_eye_y,right_eye_score,left_ear_x,left_ear_y,left_ear_score,right_ear_x,right_ear_y,right_ear_score,left_shoulder_x,left_shoulder_y,left_shoulder_score,right_shoulder_x,right_shoulder_y,right_shoulder_score,left_elbow_x,left_elbow_y,left_elbow_score,right_elbow_x,right_elbow_y,right_elbow_score,left_wrist_x,left_wrist_y,left_wrist_score,right_wrist_x,right_wrist_y,right_wrist_score,left_hip_x,left_hip_y,left_hip_score,right_hip_x,right_hip_y,right_hip_score,left_knee_x,left_knee_y,left_knee_score,right_knee_x,right_knee_y,right_knee_score,left_ankle_x,left_ankle_y,left_ankle_score,right_ankle_x,right_ankle_y,right_ankle_score\n';

    const csvContent = csvHeader + this.keypointsData.join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    document.body.appendChild(a);
    a.style = 'display: none';
    a.href = url;
    a.download = 'keypoints_with_scores.csv';
    a.click();
    window.URL.revokeObjectURL(url);

    console.log('CSV exported');
  }
}
