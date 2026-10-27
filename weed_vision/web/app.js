"use strict";

const byId = (id) => document.getElementById(id);
const form = byId("test-form");
const fileInput = byId("image-file");
const preview = byId("image-preview");
let previewUrl = null;

const errors = {
  unsupported_image_type: "图片格式不支持，请上传 JPG、PNG 或 WebP 图片。",
  image_type_mismatch: "图片内容与声明的格式不一致，请重新保存后上传。",
  empty_image: "图片文件为空，请重新选择。",
  invalid_image: "无法读取这张图片，文件可能已损坏或不完整。",
  animated_image: "暂不支持动画图片，请上传单帧图片。",
  image_too_large: "图片文件大小或像素数超过服务限制，请缩小图片后重试。",
  request_too_large: "上传内容超过服务限制，请缩小图片后重试。",
  validation_error: "请求参数不正确，请检查相机编号和采集时间戳。",
  model_unavailable: "模型尚未就绪，请检查服务终端日志并刷新模型状态。",
  inference_failed: "模型推理失败，请使用下方请求编号检查服务终端日志。",
  internal_error: "服务内部发生错误，请使用下方请求编号检查服务终端日志。",
  http_error: "请求未能完成，请确认本地服务地址并重试。",
  invalid_sensor_context: "传感器数据格式不正确，请检查时间戳字符串、坐标和字段名。",
  invalid_calibration: "标定文件无效，请检查矩阵、相机参数和标定编号。",
  image_orientation_mismatch: "映射图片方向必须与相机标定一致，请使用相机原始图像。",
  context_too_large: "传感器 JSON 超过 48 KiB 限制。",
  mapping_failed: "映射计算失败，请使用请求编号查看服务日志。",
  mapping_backend_unavailable: "当前模型后端无法提供映射需要的原始掩膜。",
};

const mappingLabels = {
  ready: "映射完成", partial: "部分映射可用", unavailable: "映射不可用",
  calibration_missing: "未提供或未找到标定", camera_mismatch: "相机编号不匹配",
  image_size_mismatch: "图像尺寸与标定不匹配", image_space_mismatch: "图像去畸变状态不匹配",
  ground_plane_unavailable: "地面平面缺失或当前未确认有效", synthetic_calibration: "正在使用模拟标定",
  radar_missing: "未提供雷达数据", radar_calibration_missing: "未提供雷达外参",
  radar_frame_mismatch: "雷达编号或坐标系不匹配", clock_mismatch: "时钟来源不一致", time_skew: "时间差超限",
  radar_clock_mismatch: "雷达时钟来源不一致", radar_time_skew: "雷达时间差超限",
  altitude_clock_mismatch: "海拔时钟来源不一致", altitude_time_skew: "海拔时间差超限",
  altitude_metadata_only: "海拔仅记录，未作为深度", insufficient_geometry: "缺少三维测量",
  candidate: "单一候选关联", ambiguous: "多个候选，关联不确定", unassociated: "未关联识别实例",
  behind_camera: "相机后方", outside_image: "图像范围外", out_of_range: "超过工作距离",
  invalid_projection: "投影无效", invalid_ground_intersection: "射线无法得到有效地面交点",
  estimated: "地面平面估计", empty_mask: "掩膜为空",
};
const mappingLabel = (key) => mappingLabels[key] || key;
byId("sensor-context").value = JSON.stringify({ camera_id: "front_rgb", capture_timestamp_ns: "0", clock_id: "jetson", image_space: "raw", calibration_id: null, ground_plane_valid: false }, null, 2);
byId("mapping-mode").addEventListener("change", () => {
  const enabled = byId("mapping-mode").checked;
  byId("mapping-fields").hidden = !enabled;
  byId("camera-id").disabled = enabled;
  byId("capture-timestamp").disabled = enabled;
  clearResult();
});

function feedback(message, kind = "") {
  byId("feedback").textContent = message;
  byId("feedback").className = `feedback ${kind}`;
}

function clearResult() {
  byId("result-content").hidden = true;
  byId("result-empty").hidden = false;
  byId("raw-section").hidden = true;
  byId("raw-section").open = false;
  byId("raw-response").textContent = "";
  byId("detections-body").replaceChildren();
  byId("mapping-results").hidden = true;
  byId("radar-body").replaceChildren();
}

async function checkHealth() {
  const button = byId("check-health");
  button.disabled = true;
  byId("health-status").textContent = "正在检查模型状态…";
  try {
    const response = await fetch("/health", { cache: "no-store", signal: AbortSignal.timeout(15000) });
    const health = await response.json();
    byId("health-status").textContent = response.ok && health.model_ready ? "模型已就绪，可以开始识别" : "模型未就绪，请检查服务终端日志";
    byId("model-version").textContent = health.model_version ? `模型版本：${health.model_version}` : "暂时无法获取模型版本";
  } catch {
    byId("health-status").textContent = "无法连接服务，请确认服务已经启动";
    byId("model-version").textContent = "启动服务后，点击“刷新状态”重试。";
  } finally {
    button.disabled = false;
  }
}

fileInput.addEventListener("change", () => {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  clearResult();
  preview.hidden = true;
  preview.removeAttribute("src");
  byId("preview-empty").hidden = false;
  byId("preview-empty").textContent = "选择图片后，这里将显示原图。";
  const file = fileInput.files[0];
  byId("file-name").textContent = file ? file.name : "尚未选择图片";
  feedback(file ? "图片已选择，点击“开始识别”进行测试。" : "选择图片后即可开始测试。");
  if (!file) return;
  previewUrl = URL.createObjectURL(file);
  preview.onload = () => { preview.hidden = false; byId("preview-empty").hidden = true; };
  preview.onerror = () => { preview.hidden = true; byId("preview-empty").textContent = "无法预览此文件，请检查图片格式。"; };
  preview.src = previewUrl;
});

function showResult(payload, timestamp) {
  byId("result-empty").hidden = true;
  byId("result-content").hidden = false;
  byId("result-count").textContent = `${payload.detections.length} 个`;
  byId("result-time").textContent = `${Number(payload.inference_ms).toFixed(1)} 毫秒`;
  byId("result-size").textContent = `${payload.image_width} × ${payload.image_height}`;
  byId("result-context").textContent = `相机编号：${payload.camera_id} · 采集时间戳：${timestamp || "未提供"}`;
  byId("result-request").textContent = `请求编号：${payload.request_id}`;
  byId("result-model").textContent = `模型版本：${payload.model_version}`;
  byId("no-detections").hidden = payload.detections.length !== 0;
  byId("detection-table-wrapper").hidden = payload.detections.length === 0;
  const rows = payload.detections.map((detection, index) => {
    const row = document.createElement("tr");
    const values = [
      detection.class_name,
      `${(detection.confidence * 100).toFixed(1)}%`,
      detection.bbox_xyxy.map((v) => Number(v).toFixed(1)).join(", "),
      detection.centroid_px.map((v) => Number(v).toFixed(1)).join(", "),
      String(detection.area_px),
      payload.mapping ? (payload.mapping.detections[index]?.ground_position_m ? JSON.stringify(payload.mapping.detections[index].ground_position_m.map(v => Number(v.toFixed(3)))) : mappingLabel(payload.mapping.detections[index]?.status || "unavailable")) : "未标定",
    ];
    for (const value of values) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.appendChild(cell);
    }
    return row;
  });
  byId("detections-body").replaceChildren(...rows);
  if (payload.mapping) {
    const mapping = payload.mapping;
    byId("mapping-results").hidden = false;
    byId("mapping-status").textContent = `${mappingLabel(mapping.status)}；坐标系：${mapping.output_frame_id}；${mapping.reasons.map(mappingLabel).join("；")}${mapping.altitude ? `；海拔记录：${mapping.altitude.value_m} 米（${mapping.altitude.reference}）` : ""}`;
    const canvas = byId("mapping-canvas");
    canvas.width = payload.image_width; canvas.height = payload.image_height;
    const ctx = canvas.getContext("2d");
    if (preview.complete && preview.naturalWidth) ctx.drawImage(preview, 0, 0, canvas.width, canvas.height);
    const scale = Math.max(1, canvas.width / 700);
    ctx.fillStyle = "#0073ff"; ctx.strokeStyle = "white"; ctx.lineWidth = 2 * scale; ctx.font = `${16 * scale}px sans-serif`;
    mapping.radar_targets.forEach((point, index) => {
      if (point.pixel_px && point.status !== "outside_image") {
        const [x, y] = point.pixel_px;
        ctx.beginPath(); ctx.arc(x, y, 6 * scale, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
        ctx.fillText(String(index + 1), x + 9 * scale, y);
      }
      const row = document.createElement("tr");
      [`${index + 1}. ${point.target_id}`, mappingLabel(point.status), point.pixel_px?.map(v => v.toFixed(1)).join(", ") || "—", point.radar_position_base_m?.map(v => v.toFixed(3)).join(", ") || "—", point.candidate_detection_indices.map(i => i + 1).join(", ") || "—"].forEach(value => {
        const cell = document.createElement("td"); cell.textContent = value; row.appendChild(cell);
      });
      byId("radar-body").appendChild(row);
    });
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearResult();
  const file = fileInput.files[0];
  const camera = byId("camera-id").value;
  const timestamp = byId("capture-timestamp").value.trim();
  if (!file) { feedback("请先选择一张图片。", "error"); fileInput.focus(); return; }
  const mappingMode = byId("mapping-mode").checked;
  if (!mappingMode && !/^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$/.test(camera)) {
    feedback("相机编号须以英文字母或数字开头，最多 64 个字符，只允许字母、数字和 _ . : -。", "error");
    byId("camera-id").focus(); return;
  }
  // Keep timestamps as decimal strings throughout the form; Number loses nanosecond precision.
  if (!mappingMode && timestamp && (!/^\d+$/.test(timestamp) || BigInt(timestamp) > 9223372036854775807n)) {
    feedback("采集时间戳须为 0 到 9223372036854775807 之间的整数，或留空。", "error");
    byId("capture-timestamp").focus(); return;
  }
  const mimeByExtension = { jpg: "image/jpeg", jpeg: "image/jpeg", png: "image/png", webp: "image/webp" };
  const mime = file.type || mimeByExtension[file.name.split(".").pop().toLowerCase()];
  if (!["image/jpeg", "image/png", "image/webp"].includes(mime)) {
    feedback(errors.unsupported_image_type, "error"); return;
  }
  const data = new FormData();
  data.append("file", file.type ? file : new File([file], file.name, { type: mime }));
  data.append("camera_id", camera);
  if (timestamp) data.append("capture_timestamp_ns", timestamp);
  if (mappingMode) {
    const rawContext = byId("sensor-context").value;
    try { JSON.parse(rawContext); } catch { feedback("传感器数据不是有效 JSON。", "error"); return; }
    // Send original text, never JSON.stringify parsed nanosecond numbers.
    data.append("context", rawContext);
  }
  byId("form-fields").disabled = true;
  byId("submit-button").textContent = "正在识别…";
  byId("results-panel").setAttribute("aria-busy", "true");
  feedback("正在上传并识别，请稍候…");
  try {
    const response = await fetch(mappingMode ? "/v1/scene-mapping" : "/v1/weed-segmentation", { method: "POST", body: data, signal: AbortSignal.timeout(180000) });
    const raw = await response.text();
    const payload = JSON.parse(raw);
    // Display unmodified response text, preserving the exact JSON integer timestamp.
    byId("raw-response").textContent = raw;
    byId("raw-section").hidden = false;
    if (!response.ok) {
      const message = errors[payload.error?.code] || `请求失败（状态码 ${response.status}），请检查服务终端日志。`;
      feedback(`${message}${payload.request_id ? ` 请求编号：${payload.request_id}` : ""}`, "error");
      byId("raw-section").open = true;
      if (response.status === 503) checkHealth();
      return;
    }
    if (!Array.isArray(payload.detections)) throw new Error("invalid_response");
    showResult(payload, mappingMode ? payload.capture_timestamp_ns : timestamp);
    feedback(`识别完成，共返回 ${payload.detections.length} 个实例。`, "success");
  } catch (error) {
    feedback(error.name === "TimeoutError" ? "等待结果超时。可以缩小图片后重试，或查看服务终端日志。" : "未能获取有效结果，请确认服务正在运行，然后重试。", "error");
  } finally {
    byId("form-fields").disabled = false;
    byId("submit-button").textContent = "开始识别";
    byId("results-panel").setAttribute("aria-busy", "false");
  }
});

byId("check-health").addEventListener("click", checkHealth);
checkHealth();
