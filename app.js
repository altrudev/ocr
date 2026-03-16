/**
 * ALTRU OCR — Complete Application
 * By ALTRU.dev — Code for Humanity
 *
 * Self-contained: preprocessing pipeline + PaddleOCR ONNX inference +
 * layout analysis + confidence calibration + reactive UI.
 * Only external dependency: onnxruntime-web (loaded from CDN).
 */

/* ═══════════════════════════════════════════════════════════════
   SECTION 1: PREPROCESSING PIPELINE
   ═══════════════════════════════════════════════════════════════ */

function createCanvas(w, h) {
  if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(w, h);
  const c = document.createElement('canvas'); c.width = w; c.height = h; return c;
}

function getCtx(canvas) {
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Failed to get 2D context');
  return ctx;
}

function clamp(v, lo = 0, hi = 255) { return v < lo ? lo : v > hi ? hi : v; }
function lum(r, g, b) { return 0.299 * r + 0.587 * g + 0.114 * b; }

function imageDataFromSource(source) {
  if (source instanceof ImageData) return source;
  const w = source.naturalWidth || source.width;
  const h = source.naturalHeight || source.height;
  const c = createCanvas(w, h);
  const ctx = getCtx(c);
  ctx.drawImage(source, 0, 0);
  return ctx.getImageData(0, 0, w, h);
}

function imageDataToCanvas(imageData) {
  const c = createCanvas(imageData.width, imageData.height);
  getCtx(c).putImageData(imageData, 0, 0);
  return c;
}

// ── Grayscale ──
function toGrayscale(imageData) {
  const { data, width, height } = imageData;
  const out = new Uint8ClampedArray(data.length);
  for (let i = 0; i < data.length; i += 4) {
    const g = lum(data[i], data[i+1], data[i+2]);
    out[i] = out[i+1] = out[i+2] = g;
    out[i+3] = data[i+3];
  }
  return new ImageData(out, width, height);
}

// ── CLAHE ──
function clahe(imageData, tileSize = 8, clipLimit = 2.0) {
  const gray = toGrayscale(imageData);
  const { data, width, height } = gray;
  const out = new Uint8ClampedArray(data.length);
  const tilesX = Math.max(1, Math.round(width / tileSize));
  const tilesY = Math.max(1, Math.round(height / tileSize));
  const tw = width / tilesX, th = height / tilesY;

  const cdfs = [];
  for (let ty = 0; ty < tilesY; ty++) {
    cdfs[ty] = [];
    for (let tx = 0; tx < tilesX; tx++) {
      const hist = new Float64Array(256);
      const x0 = Math.round(tx * tw), y0 = Math.round(ty * th);
      const x1 = Math.round((tx+1) * tw), y1 = Math.round((ty+1) * th);
      const area = Math.max(1, (x1-x0) * (y1-y0));
      for (let y = y0; y < y1 && y < height; y++)
        for (let x = x0; x < x1 && x < width; x++)
          hist[data[(y*width+x)*4]]++;
      const limit = Math.max(1, clipLimit * (area/256));
      let excess = 0;
      for (let i = 0; i < 256; i++) { if (hist[i]>limit) { excess+=hist[i]-limit; hist[i]=limit; } }
      const inc = excess/256;
      for (let i = 0; i < 256; i++) hist[i] += inc;
      const cdf = new Float64Array(256);
      cdf[0] = hist[0];
      for (let i = 1; i < 256; i++) cdf[i] = cdf[i-1]+hist[i];
      const cMin = cdf[0], cMax = cdf[255], den = Math.max(1, cMax-cMin);
      for (let i = 0; i < 256; i++) cdf[i] = Math.round(((cdf[i]-cMin)/den)*255);
      cdfs[ty][tx] = cdf;
    }
  }
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const pi = (y*width+x)*4, v = data[pi];
      const ftx = (x/tw)-0.5, fty = (y/th)-0.5;
      const tx0 = clamp(Math.floor(ftx),0,tilesX-1), ty0 = clamp(Math.floor(fty),0,tilesY-1);
      const tx1 = Math.min(tx0+1,tilesX-1), ty1 = Math.min(ty0+1,tilesY-1);
      const fx = Math.max(0,Math.min(1,ftx-tx0)), fy = Math.max(0,Math.min(1,fty-ty0));
      const r = (1-fy)*((1-fx)*cdfs[ty0][tx0][v]+fx*cdfs[ty0][tx1][v])+fy*((1-fx)*cdfs[ty1][tx0][v]+fx*cdfs[ty1][tx1][v]);
      out[pi]=out[pi+1]=out[pi+2]=clamp(Math.round(r)); out[pi+3]=255;
    }
  }
  return new ImageData(out, width, height);
}

// ── Sauvola Binarization ──
function sauvola(imageData, windowSize = 15, k = 0.2, R = 128) {
  const { width, height } = imageData;
  const gray = toGrayscale(imageData);
  const src = gray.data;
  const out = new Uint8ClampedArray(src.length);
  const n = width * height;
  const integral = new Float64Array(n), integralSq = new Float64Array(n);
  for (let y = 0; y < height; y++) {
    let rs = 0, rss = 0;
    for (let x = 0; x < width; x++) {
      const idx = y*width+x, v = src[idx*4];
      rs += v; rss += v*v;
      integral[idx] = rs + (y>0 ? integral[(y-1)*width+x] : 0);
      integralSq[idx] = rss + (y>0 ? integralSq[(y-1)*width+x] : 0);
    }
  }
  const half = Math.floor(windowSize/2);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const y1=Math.max(0,y-half)-1, y2=Math.min(height-1,y+half);
      const x1=Math.max(0,x-half)-1, x2=Math.min(width-1,x+half);
      const area=(y2-(y1<0?-1:y1))*(x2-(x1<0?-1:x1));
      let sum=integral[y2*width+x2], ssq=integralSq[y2*width+x2];
      if(y1>=0){sum-=integral[y1*width+x2];ssq-=integralSq[y1*width+x2];}
      if(x1>=0){sum-=integral[y2*width+x1];ssq-=integralSq[y2*width+x1];}
      if(y1>=0&&x1>=0){sum+=integral[y1*width+x1];ssq+=integralSq[y1*width+x1];}
      const mean=sum/area, std=Math.sqrt(Math.max(0,(ssq/area)-mean*mean));
      const thresh=mean*(1+k*(std/R-1));
      const pi=(y*width+x)*4, v=src[pi]>thresh?255:0;
      out[pi]=out[pi+1]=out[pi+2]=v; out[pi+3]=255;
    }
  }
  return new ImageData(out, width, height);
}

// ── Deskew ──
function estimateSkewAngle(imageData, range = 10, steps = 40) {
  const gray = toGrayscale(imageData);
  const { width, height } = gray;
  const scale = Math.min(1, 400 / Math.max(width, height));
  const sw = Math.round(width*scale), sh = Math.round(height*scale);
  const tmp = createCanvas(width, height);
  getCtx(tmp).putImageData(gray, 0, 0);
  const tc = createCanvas(sw, sh);
  const ctx = getCtx(tc);
  let bestAngle = 0, bestVar = -1;
  for (let i = 0; i <= steps; i++) {
    const angle = -range + (2*range*i)/steps;
    const rad = (angle*Math.PI)/180;
    ctx.clearRect(0,0,sw,sh);
    ctx.save(); ctx.translate(sw/2,sh/2); ctx.rotate(rad); ctx.drawImage(tmp,-sw/2,-sh/2,sw,sh); ctx.restore();
    const rot = ctx.getImageData(0,0,sw,sh);
    const hProj = new Float64Array(sh);
    for (let y=0;y<sh;y++) { let s=0; for(let x=0;x<sw;x++) s+=(255-rot.data[(y*sw+x)*4]); hProj[y]=s; }
    let mean=0; for(let y=0;y<sh;y++) mean+=hProj[y]; mean/=sh;
    let v=0; for(let y=0;y<sh;y++) v+=(hProj[y]-mean)**2; v/=sh;
    if(v>bestVar){bestVar=v;bestAngle=angle;}
  }
  return bestAngle;
}

function deskew(imageData, angle) {
  if (angle === undefined || angle === null) angle = estimateSkewAngle(imageData);
  if (Math.abs(angle) < 0.15) return imageData;
  const { width, height } = imageData;
  const rad = (-angle*Math.PI)/180;
  const cos=Math.abs(Math.cos(rad)), sin=Math.abs(Math.sin(rad));
  const nw=Math.round(width*cos+height*sin), nh=Math.round(width*sin+height*cos);
  const src = createCanvas(width, height); getCtx(src).putImageData(imageData, 0, 0);
  const dst = createCanvas(nw, nh); const ctx = getCtx(dst);
  ctx.fillStyle='#FFF'; ctx.fillRect(0,0,nw,nh);
  ctx.translate(nw/2,nh/2); ctx.rotate(rad); ctx.drawImage(src,-width/2,-height/2);
  return ctx.getImageData(0,0,nw,nh);
}

// ── Median Filter ──
function medianFilter(imageData, radius = 1) {
  const { data, width, height } = imageData;
  const out = new Uint8ClampedArray(data.length);
  const size = (2*radius+1)**2;
  const buf = new Uint8Array(size);
  for (let y=0;y<height;y++) for(let x=0;x<width;x++) {
    for(let ch=0;ch<3;ch++) {
      let cnt=0;
      for(let dy=-radius;dy<=radius;dy++) for(let dx=-radius;dx<=radius;dx++) {
        const ny=clamp(y+dy,0,height-1), nx=clamp(x+dx,0,width-1);
        buf[cnt++]=data[(ny*width+nx)*4+ch];
      }
      buf.subarray(0,cnt).sort();
      out[(y*width+x)*4+ch]=buf[cnt>>1];
    }
    out[(y*width+x)*4+3]=255;
  }
  return new ImageData(out, width, height);
}

// ── Scale Normalization ──
function normalizeScale(imageData, minH = 800, maxH = 2500) {
  const { width, height } = imageData;
  if (height >= minH && height <= maxH) return { imageData, factor: 1 };
  const scale = height < minH ? minH/height : maxH/height;
  const nw = Math.round(width*scale), nh = Math.round(height*scale);
  const src = createCanvas(width,height); getCtx(src).putImageData(imageData,0,0);
  const dst = createCanvas(nw,nh); const ctx = getCtx(dst);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(src,0,0,nw,nh);
  return { imageData: ctx.getImageData(0,0,nw,nh), factor: scale };
}

// ── Composite Pipeline ──
const PRESETS = {
  auto:        { clahe:true,  deskew:true,  denoise:false, binarize:false, claheClip:2.0 },
  cleanScan:   { clahe:false, deskew:true,  denoise:false, binarize:false, claheClip:2.0 },
  phonePhoto:  { clahe:true,  deskew:true,  denoise:true,  binarize:false, claheClip:3.0 },
  degradedScan:{ clahe:true,  deskew:true,  denoise:true,  binarize:true,  claheClip:3.0 },
  receipt:     { clahe:true,  deskew:true,  denoise:false, binarize:true,  claheClip:4.0 },
};

function preprocess(imageData, preset = 'auto') {
  const p = PRESETS[preset] || PRESETS.auto;
  const meta = { skewAngle: null, scaled: false, factor: 1, denoised: false, binarized: false };
  let img = imageData;

  // Scale
  const sc = normalizeScale(img);
  img = sc.imageData; meta.factor = sc.factor; meta.scaled = sc.factor !== 1;

  // CLAHE
  if (p.clahe) img = clahe(img, 8, p.claheClip);

  // Grayscale
  img = toGrayscale(img);

  // Deskew
  if (p.deskew) {
    const angle = estimateSkewAngle(img);
    if (Math.abs(angle) >= 0.15) { img = deskew(img, angle); meta.skewAngle = angle; }
  }

  // Denoise
  if (p.denoise) { img = medianFilter(img, 1); meta.denoised = true; }

  // Binarize
  if (p.binarize) { img = sauvola(img, 15, 0.2); meta.binarized = true; }

  return { imageData: img, meta };
}


/* ═══════════════════════════════════════════════════════════════
   SECTION 2: PADDLEOCR ONNX INFERENCE (VENDORED)
   ═══════════════════════════════════════════════════════════════ */

const DEFAULT_MODELS = {
  det: 'https://raw.githubusercontent.com/niceandbright/ppu-paddle-ocr-models/main/models/det/en_PP-OCRv3_det_infer.onnx',
  rec: 'https://raw.githubusercontent.com/niceandbright/ppu-paddle-ocr-models/main/models/rec/en_PP-OCRv4_rec_infer.onnx',
  dict:'https://raw.githubusercontent.com/niceandbright/ppu-paddle-ocr-models/main/keys/en_dict.txt',
};

class PaddleOCR {
  constructor() { this.detSession = null; this.recSession = null; this.dictionary = []; }

  async init(onProgress) {
    onProgress?.('Loading ONNX Runtime...');
    // ort is loaded globally from CDN in index.html
    const sess = ort.InferenceSession;
    const opts = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };

    onProgress?.('Downloading detection model...');
    const detBuf = await fetch(DEFAULT_MODELS.det).then(r => r.arrayBuffer());
    this.detSession = await sess.create(detBuf, opts);

    onProgress?.('Downloading recognition model...');
    const recBuf = await fetch(DEFAULT_MODELS.rec).then(r => r.arrayBuffer());
    this.recSession = await sess.create(recBuf, opts);

    onProgress?.('Loading dictionary...');
    const dictText = await fetch(DEFAULT_MODELS.dict).then(r => r.text());
    this.dictionary = dictText.split('\n').filter(l => l.length > 0);

    onProgress?.('Ready.');
  }

  async recognize(imageData) {
    const { width, height, data } = imageData;

    // ── Detection preprocessing ──
    const maxSide = 960;
    let ratio = Math.max(width,height) > maxSide ? maxSide/Math.max(width,height) : 1;
    let rw = Math.max(32, Math.round(width*ratio/32)*32);
    let rh = Math.max(32, Math.round(height*ratio/32)*32);

    const tmpC = createCanvas(width,height); getCtx(tmpC).putImageData(imageData,0,0);
    const resC = createCanvas(rw,rh); getCtx(resC).drawImage(tmpC,0,0,rw,rh);
    const resD = getCtx(resC).getImageData(0,0,rw,rh);

    const mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225];
    const px = rw*rh;
    const chw = new Float32Array(3*px);
    for(let i=0;i<px;i++) {
      const pi=i*4;
      chw[i]=(resD.data[pi]/255-mean[0])/std[0];
      chw[px+i]=(resD.data[pi+1]/255-mean[1])/std[1];
      chw[2*px+i]=(resD.data[pi+2]/255-mean[2])/std[2];
    }

    const detT = new ort.Tensor('float32', chw, [1,3,rh,rw]);
    const detF = {}; detF[this.detSession.inputNames[0]] = detT;
    const detR = await this.detSession.run(detF);
    const heatmap = detR[this.detSession.outputNames[0]].data;

    // ── Extract boxes from heatmap ──
    const boxes = this._extractBoxes(heatmap, rw, rh, width, height);
    if (boxes.length === 0) return { text: '', lines: [] };

    // ── Recognition per box ──
    const lines = [];
    for (const box of boxes) {
      const crop = this._cropForRec(imageData, box);
      if (!crop) continue;
      const { tensor, cw, ch } = crop;
      const recT = new ort.Tensor('float32', tensor, [1,3,ch,cw]);
      const recF = {}; recF[this.recSession.inputNames[0]] = recT;
      const recR = await this.recSession.run(recF);
      const recOut = recR[this.recSession.outputNames[0]];
      const decoded = this._ctcDecode(recOut.data, recOut.dims[1], recOut.dims[2]);
      if (decoded.text.length > 0) lines.push({ text: decoded.text, score: decoded.score, box });
    }

    return { text: lines.map(l=>l.text).join('\n'), lines };
  }

  _extractBoxes(heatmap, w, h, origW, origH) {
    const binThresh=0.3, boxThresh=0.6, unclipRatio=1.5;
    const binary = new Uint8Array(w*h);
    for(let i=0;i<binary.length;i++) binary[i]=heatmap[i]>binThresh?1:0;

    const boxes=[], visited=new Uint8Array(w*h);
    for(let y=1;y<h-1;y++) for(let x=1;x<w-1;x++) {
      const idx=y*w+x;
      if(!binary[idx]||visited[idx]) continue;
      let minX=x,maxX=x,minY=y,maxY=y,sumS=0,cnt=0;
      const q=[idx]; visited[idx]=1;
      while(q.length>0) {
        const ci=q.pop(), cx=ci%w, cy=(ci/w)|0;
        minX=Math.min(minX,cx); maxX=Math.max(maxX,cx);
        minY=Math.min(minY,cy); maxY=Math.max(maxY,cy);
        sumS+=heatmap[ci]; cnt++;
        for(const [dx,dy] of [[1,0],[-1,0],[0,1],[0,-1]]) {
          const nx=cx+dx, ny=cy+dy;
          if(nx<0||nx>=w||ny<0||ny>=h) continue;
          const ni=ny*w+nx;
          if(!visited[ni]&&binary[ni]) { visited[ni]=1; q.push(ni); }
        }
      }
      if(sumS/cnt<boxThresh||cnt<10) continue;
      const bw=maxX-minX, bh=maxY-minY;
      const padX=bw*(unclipRatio-1)*0.5, padY=bh*(unclipRatio-1)*0.5;
      const sx=origW/w, sy=origH/h;
      const x0=Math.max(0,(minX-padX)*sx), y0=Math.max(0,(minY-padY)*sy);
      const x1=Math.min(origW,(maxX+padX)*sx), y1=Math.min(origH,(maxY+padY)*sy);
      if(x1-x0<3||y1-y0<3) continue;
      boxes.push([[x0|0,y0|0],[x1|0,y0|0],[x1|0,y1|0],[x0|0,y1|0]]);
    }
    boxes.sort((a,b) => { const dy=a[0][1]-b[0][1]; return Math.abs(dy)>10?dy:a[0][0]-b[0][0]; });
    return boxes;
  }

  _cropForRec(imageData, box) {
    const { width, height } = imageData;
    const x0=Math.max(0,Math.min(box[0][0],box[3][0])), y0=Math.max(0,Math.min(box[0][1],box[1][1]));
    const x1=Math.min(width,Math.max(box[1][0],box[2][0])), y1=Math.min(height,Math.max(box[2][1],box[3][1]));
    const cw=Math.round(x1-x0), ch_=Math.round(y1-y0);
    if(cw<2||ch_<2) return null;
    const rh=48, rw=Math.max(1,Math.round(cw*rh/ch_));
    const tmp=createCanvas(width,height); getCtx(tmp).putImageData(imageData,0,0);
    const crop=createCanvas(rw,rh); getCtx(crop).drawImage(tmp,x0,y0,cw,ch_,0,0,rw,rh);
    const cd=getCtx(crop).getImageData(0,0,rw,rh);
    const px=rw*rh, tensor=new Float32Array(3*px);
    for(let i=0;i<px;i++){const pi=i*4;tensor[i]=(cd.data[pi]/255-0.5)/0.5;tensor[px+i]=(cd.data[pi+1]/255-0.5)/0.5;tensor[2*px+i]=(cd.data[pi+2]/255-0.5)/0.5;}
    return { tensor, cw: rw, ch: rh };
  }

  _ctcDecode(output, seqLen, vocabSize) {
    let text='', prev=0, totalS=0, sCnt=0;
    for(let t=0;t<seqLen;t++) {
      let maxI=0,maxV=-Infinity;
      for(let v=0;v<vocabSize;v++){const val=output[t*vocabSize+v];if(val>maxV){maxV=val;maxI=v;}}
      const conf=1/(1+Math.exp(-maxV));
      if(maxI!==0&&maxI!==prev&&maxI-1<this.dictionary.length){text+=this.dictionary[maxI-1];totalS+=conf;sCnt++;}
      prev=maxI;
    }
    return { text: text.trim(), score: sCnt>0?totalS/sCnt:0 };
  }

  destroy() { this.detSession?.release?.(); this.recSession?.release?.(); }
}


/* ═══════════════════════════════════════════════════════════════
   SECTION 3: LAYOUT ANALYSIS + CONFIDENCE CALIBRATION
   ═══════════════════════════════════════════════════════════════ */

function analyzeLayout(lines, imgW, imgH) {
  if (lines.length === 0) return [];
  const bboxes = lines.map(l => {
    const xs=l.box.map(p=>p[0]), ys=l.box.map(p=>p[1]);
    return { x:Math.min(...xs), y:Math.min(...ys), w:Math.max(...xs)-Math.min(...xs), h:Math.max(...ys)-Math.min(...ys) };
  });
  const heights = bboxes.map(b=>b.h).sort((a,b)=>a-b);
  const medH = heights[heights.length>>1]||20;

  return lines.map((l,i) => {
    const bb = bboxes[i];
    let type = 'paragraph';
    if (bb.y < imgH*0.08) type = 'header';
    else if (bb.y+bb.h > imgH*0.92) type = 'footer';
    else if (bb.h/medH > 1.6 && l.text.length < 100) type = 'heading';
    else if (bb.h/medH < 0.7 && l.text.length < 80) type = 'caption';

    // Confidence calibration
    const logit = Math.log(Math.max(1e-8,l.score)/Math.max(1e-8,1-l.score));
    let conf = 1/(1+Math.exp(-logit));
    const len = l.text.trim().length;
    if(len===1) conf*=0.6; else if(len===2) conf*=0.8; else if(len<=4) conf*=0.9;
    conf = Math.min(1, Math.max(0, conf));

    return { text:l.text, rawScore:l.score, confidence:conf, box:l.box, bbox:bb, type, order:i };
  }).filter(r => r.confidence >= 0.25);
}


/* ═══════════════════════════════════════════════════════════════
   SECTION 4: UI (VANILLA JS, NO FRAMEWORK)
   ═══════════════════════════════════════════════════════════════ */

const state = {
  status: 'idle',     // idle | loading | ready | processing | error
  progress: 0,
  message: '',
  error: null,
  result: null,
  preset: 'auto',
  mode: 'upload',     // upload | camera
  previewUrl: null,
  cameraStream: null,
};

let engine = null;

// ── Render ──
function render() {
  const app = document.getElementById('app');
  app.innerHTML = `
    <header class="header">
      <div class="header-inner">
        <div class="logo-group">
          <div class="logo-icon">A</div>
          <div class="logo-text"><h1>ALTRU OCR</h1><p>Code for Humanity</p></div>
        </div>
        <div>
          <span class="status-dot ${state.status}"></span>
          <span style="font-size:0.7rem;color:rgba(255,255,255,0.4)">${state.status}</span>
        </div>
        <div class="mode-toggle">
          <button class="mode-btn ${state.mode==='upload'?'active':''}" onclick="setMode('upload')">Upload</button>
          <button class="mode-btn ${state.mode==='camera'?'active':''}" onclick="setMode('camera')">Camera</button>
        </div>
      </div>
    </header>
    <main class="main">
      ${renderPresets()}
      ${state.error ? `<div class="error-msg">${state.error}</div>` : ''}
      ${state.status==='loading'||state.status==='processing' ? renderProgress() : ''}
      <div class="results-grid ${state.result ? 'has-result' : ''}">
        <div>
          ${state.mode==='upload' ? renderUpload() : renderCamera()}
          ${state.previewUrl ? `<div class="preview glass" style="margin-top:1rem;overflow:hidden"><img src="${state.previewUrl}" alt="Preview"></div>` : ''}
        </div>
        ${state.result ? renderResult() : ''}
      </div>
      <div class="footer">All processing runs locally in your browser. No data leaves your device.</div>
    </main>
  `;
  attachEvents();
}

function renderPresets() {
  const labels = { auto:'Auto', cleanScan:'Clean Scan', phonePhoto:'Phone Photo', degradedScan:'Old/Degraded', receipt:'Receipt' };
  return `<div class="presets">
    <span class="presets-label">Preset:</span>
    ${Object.keys(PRESETS).map(p => `<button class="preset-btn ${state.preset===p?'active':''}" data-preset="${p}">${labels[p]}</button>`).join('')}
  </div>`;
}

function renderProgress() {
  return `<div class="progress-bar glass">
    <div class="meta"><span class="label">${state.message}</span><span class="pct">${Math.round(state.progress*100)}%</span></div>
    <div class="progress-track"><div class="progress-fill" style="width:${Math.round(state.progress*100)}%"></div></div>
  </div>`;
}

function renderUpload() {
  return `<div class="upload-zone glass" id="upload-zone">
    <input type="file" id="file-input" accept="image/*">
    <div class="upload-icon"><svg width="24" height="24" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5" style="color:var(--teal)"><path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"/></svg></div>
    <p>Drop an image or click to upload</p>
    <p class="subtext">JPG, PNG, BMP, WebP — processed locally</p>
  </div>`;
}

function renderCamera() {
  if (state.cameraStream) {
    return `<div class="camera-active glass">
      <video id="camera-video" autoplay playsinline muted></video>
      <div class="camera-controls">
        <button class="cam-btn-sm" onclick="stopCamera()"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12"/></svg></button>
        <button class="shutter-btn" onclick="capturePhoto()"><div class="shutter-ring"></div></button>
      </div>
    </div>`;
  }
  return `<div class="camera-start glass" onclick="startCamera()">
    <div class="cam-icon"><svg width="24" height="24" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5" style="color:var(--coral)"><path stroke-linecap="round" stroke-linejoin="round" d="M6.827 6.175A2.31 2.31 0 015.186 7.23c-.38.054-.757.112-1.134.175C2.999 7.58 2.25 8.507 2.25 9.574V18a2.25 2.25 0 002.25 2.25h15A2.25 2.25 0 0021.75 18V9.574c0-1.067-.75-1.994-1.802-2.169a47.865 47.865 0 00-1.134-.175 2.31 2.31 0 01-1.64-1.055l-.822-1.316a2.192 2.192 0 00-1.736-1.039 48.774 48.774 0 00-5.232 0 2.192 2.192 0 00-1.736 1.039l-.821 1.316z"/><path stroke-linecap="round" stroke-linejoin="round" d="M16.5 12.75a4.5 4.5 0 11-9 0 4.5 4.5 0 019 0z"/></svg></div>
    <p style="font-size:0.875rem;color:rgba(255,255,255,0.7);font-weight:500">Tap to open camera</p>
    <p class="subtext" style="margin-top:0.375rem">Point at text, then capture</p>
  </div>`;
}

function renderResult() {
  const r = state.result;
  const tab = state._tab || 'text';
  let content = '';

  if (tab === 'text') {
    content = `<div class="result-text">${r.text || '(no text detected)'}</div>`;
  } else if (tab === 'regions') {
    if (r.regions.length === 0) {
      content = '<p style="text-align:center;color:rgba(255,255,255,0.3);padding:2rem">No regions detected.</p>';
    } else {
      content = r.regions.map(reg => {
        const badgeCls = 'badge-' + (reg.type||'unknown');
        const confCls = reg.confidence>=0.8?'conf-high':reg.confidence>=0.5?'conf-mid':'conf-low';
        return `<div class="region-card">
          <div class="region-header">
            <span class="region-badge ${badgeCls}">${reg.type}</span>
            <span class="region-order">#${reg.order}</span>
            <span class="region-conf ${confCls}">${(reg.confidence*100).toFixed(0)}%</span>
          </div>
          <p class="region-text">${reg.text}</p>
        </div>`;
      }).join('');
    }
  } else if (tab === 'meta') {
    const m = r.meta;
    content = `<div class="meta-grid">
      <span class="meta-label">Processing time</span><span class="meta-value highlight">${r.elapsed.toFixed(0)}ms</span>
      <span class="meta-label">Image size</span><span class="meta-value">${r.imageWidth}×${r.imageHeight}</span>
      <span class="meta-label">Regions found</span><span class="meta-value">${r.regions.length}</span>
      <span class="meta-label">Skew detected</span><span class="meta-value">${m.skewAngle?.toFixed(2) ?? 'none'}°</span>
      <span class="meta-label">Scaled</span><span class="meta-value">${m.scaled ? m.factor.toFixed(2)+'×' : 'no'}</span>
      <span class="meta-label">Denoised</span><span class="meta-value">${m.denoised?'yes':'no'}</span>
      <span class="meta-label">Binarized</span><span class="meta-value">${m.binarized?'yes':'no'}</span>
    </div>`;
  }

  return `<div class="result-panel glass">
    <div class="result-tabs">
      <button class="tab-btn ${tab==='text'?'active':''}" onclick="setTab('text')">Text</button>
      <button class="tab-btn ${tab==='regions'?'active':''}" onclick="setTab('regions')">Regions (${r.regions.length})</button>
      <button class="tab-btn ${tab==='meta'?'active':''}" onclick="setTab('meta')">Meta</button>
    </div>
    <div class="result-content">${content}</div>
    <div class="result-actions">
      <button class="btn-primary" onclick="copyResult()">Copy Text</button>
    </div>
  </div>`;
}

// ── Event attachment ──
function attachEvents() {
  // Presets
  document.querySelectorAll('.preset-btn').forEach(btn => {
    btn.addEventListener('click', () => { state.preset = btn.dataset.preset; render(); });
  });

  // Upload zone
  const zone = document.getElementById('upload-zone');
  const fileInput = document.getElementById('file-input');
  if (zone && fileInput) {
    zone.addEventListener('click', () => fileInput.click());
    zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('dragging'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('dragging'));
    zone.addEventListener('drop', (e) => { e.preventDefault(); zone.classList.remove('dragging'); const f=e.dataTransfer?.files[0]; if(f&&f.type.startsWith('image/')) processFile(f); });
    fileInput.addEventListener('change', (e) => { const f=e.target.files[0]; if(f) processFile(f); });
  }

  // Camera video
  const video = document.getElementById('camera-video');
  if (video && state.cameraStream) {
    video.srcObject = state.cameraStream;
    video.play();
  }
}

// ── Actions ──
window.setMode = (m) => { state.mode = m; if(m==='upload' && state.cameraStream) stopCamera(); render(); };
window.setTab = (t) => { state._tab = t; render(); };

window.copyResult = async () => {
  if (!state.result?.text) return;
  try { await navigator.clipboard.writeText(state.result.text); } catch {}
  const btn = document.querySelector('.btn-primary');
  if(btn) { btn.textContent = 'Copied!'; setTimeout(()=>{ btn.textContent='Copy Text'; }, 1500); }
};

window.startCamera = async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode:'environment', width:{ideal:1920}, height:{ideal:1080} }, audio:false });
    state.cameraStream = stream;
    render();
  } catch(e) {
    state.error = e.name==='NotAllowedError' ? 'Camera permission denied.' : 'Camera not available.';
    render();
  }
};

window.stopCamera = () => {
  if(state.cameraStream) { state.cameraStream.getTracks().forEach(t=>t.stop()); state.cameraStream = null; }
  render();
};

window.capturePhoto = () => {
  const video = document.getElementById('camera-video');
  if(!video) return;
  const c = createCanvas(video.videoWidth, video.videoHeight);
  getCtx(c).drawImage(video, 0, 0);
  stopCamera();
  c.toBlob(blob => { state.previewUrl = URL.createObjectURL(blob); render(); });
  processImageData(getCtx(c).getImageData(0,0,c.width,c.height));
};

async function processFile(file) {
  state.previewUrl = URL.createObjectURL(file);
  state.error = null;
  render();

  const img = new Image();
  img.src = state.previewUrl;
  await new Promise((res,rej) => { img.onload=res; img.onerror=rej; });

  const c = createCanvas(img.naturalWidth, img.naturalHeight);
  getCtx(c).drawImage(img, 0, 0);
  const imageData = getCtx(c).getImageData(0, 0, c.width, c.height);
  processImageData(imageData);
}

async function processImageData(imageData) {
  const start = performance.now();

  // Init engine if needed
  if (!engine) {
    state.status = 'loading'; state.progress = 0; render();
    engine = new PaddleOCR();
    await engine.init((msg) => {
      state.message = msg;
      state.progress = Math.min(0.9, state.progress + 0.2);
      render();
    });
  }

  state.status = 'processing'; state.progress = 0; state.message = 'Preprocessing...'; render();
  const origW = imageData.width, origH = imageData.height;

  // Preprocess
  const { imageData: processed, meta } = preprocess(imageData, state.preset);
  state.progress = 0.4; state.message = 'Detecting text...'; render();

  // OCR
  const result = await engine.recognize(processed);
  state.progress = 0.8; state.message = 'Analyzing layout...'; render();

  // Layout + confidence
  const regions = analyzeLayout(result.lines, processed.width, processed.height);

  const elapsed = performance.now() - start;
  state.result = {
    text: regions.map(r=>r.text).join('\n'),
    regions,
    elapsed,
    meta,
    imageWidth: origW,
    imageHeight: origH,
  };
  state._tab = 'text';
  state.status = 'ready';
  state.message = `Done in ${elapsed.toFixed(0)}ms`;
  render();
}

// ── Boot ──
document.addEventListener('DOMContentLoaded', () => {
  render();
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('./sw.js').catch(() => {});
  }
});
