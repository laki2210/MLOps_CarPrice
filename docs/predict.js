let modelData = null;
let mlModel = null;

const form = document.getElementById("prediction-form");
const statusPill = document.getElementById("status-pill");
const predictionValue = document.getElementById("prediction-value");
const predictionText = document.getElementById("prediction-text");
const barFill = document.getElementById("bar-fill");

const detailModelType = document.getElementById("detail-model-type");
const detailTarget = document.getElementById("detail-target");
const detailDataset = document.getElementById("detail-dataset");

const metricMae = document.getElementById("metric-mae");
const metricRmse = document.getElementById("metric-rmse");
const metricR2 = document.getElementById("metric-r2");

function formatMoney(value, currencySymbol = "$") {
  const rounded = Math.round(value);
  return `${currencySymbol}${rounded.toLocaleString("en-US")}`;
}

function createField(field) {
  const wrapper = document.createElement("label");
  wrapper.className = "field";

  const labelText = document.createElement("span");
  labelText.textContent = field.label;
  wrapper.appendChild(labelText);

  if (field.type === "number") {
    const input = document.createElement("input");
    input.type = "number";
    input.name = field.name;
    input.id = field.name;
    input.min = field.min;
    input.max = field.max;
    input.step = field.step;
    input.value = field.default;
    wrapper.appendChild(input);
  } else if (field.type === "select") {
    const select = document.createElement("select");
    select.name = field.name;
    select.id = field.name;

    field.options.forEach(optionValue => {
      const option = document.createElement("option");
      option.value = optionValue;
      option.textContent = optionValue;
      if (optionValue === field.default) {
        option.selected = true;
      }
      select.appendChild(option);
    });

    wrapper.appendChild(select);
  }

  return wrapper;
}

function buildForm(schema) {
  form.innerHTML = "";

  schema.forEach(field => {
    form.appendChild(createField(field));
  });

  const button = document.createElement("button");
  button.type = "submit";
  button.textContent = "Predict price";
  form.appendChild(button);
}

function readRawInput() {
  const raw = {};

  modelData.input_schema.forEach(field => {
    const element = document.getElementById(field.name);
    if (!element) return;

    if (field.type === "number") {
      raw[field.name] = Number(element.value);
    } else {
      raw[field.name] = element.value;
    }
  });

  return raw;
}

function transformInput(raw) {
  const vector = [];

  for (const feature of modelData.numeric_features) {
    let value = raw[feature];

    if (!Number.isFinite(value)) {
      value = modelData.numeric_imputers[feature];
    }

    const mean = modelData.numeric_means[feature];
    const scale = modelData.numeric_scales[feature] || 1;
    vector.push((value - mean) / scale);
  }

  for (const feature of modelData.categorical_features) {
    let value = raw[feature];

    if (value === undefined || value === null || value === "") {
      value = modelData.categorical_fill[feature];
    }

    const levels = modelData.categorical_levels[feature];
    levels.forEach(level => {
      vector.push(String(value) === String(level) ? 1 : 0);
    });
  }

  return vector;
}

function buildMlJsModel() {
  const X = modelData.browser_training_X;
  const y = modelData.browser_training_y;

  mlModel = new ML.MultivariateLinearRegression(X, y, {
    intercept: true,
    statistics: true
  });
}

function predictWithMlJs(raw) {
  const transformed = transformInput(raw);
  const prediction = mlModel.predict(transformed);

  if (Array.isArray(prediction)) {
    return Array.isArray(prediction[0]) ? prediction[0][0] : prediction[0];
  }

  return prediction;
}

function renderStaticInfo() {
  detailModelType.textContent = modelData.model_type + " (ML.js)";
  detailTarget.textContent = modelData.target_name;
  detailDataset.textContent = modelData.dataset_file;

  metricMae.textContent = modelData.test_metrics.MAE;
  metricRmse.textContent = modelData.test_metrics.RMSE;
  metricR2.textContent = modelData.test_metrics.R2;
}

function renderPrediction(value) {
  const currencySymbol = modelData.currency_symbol || "$";
  predictionValue.textContent = formatMoney(value, currencySymbol);

  const relativeWidth = Math.max(5, Math.min(100, (value / 100000) * 100));
  barFill.style.width = `${relativeWidth}%`;

  statusPill.textContent = "Prediction ready";
  statusPill.className = "pill positive";
  predictionText.textContent =
    "The value above was predicted in the browser using ML.js and the exported training representation of the linear regression workflow.";
}

form.addEventListener("submit", event => {
  event.preventDefault();
  if (!modelData || !mlModel) return;

  const raw = readRawInput();
  const prediction = predictWithMlJs(raw);
  renderPrediction(prediction);
});

fetch("model.json")
  .then(response => response.json())
  .then(data => {
    modelData = data;

    buildForm(modelData.input_schema);
    buildMlJsModel();
    renderStaticInfo();

    statusPill.textContent = "Model loaded";
    statusPill.className = "pill neutral";

    const initialPrediction = predictWithMlJs(readRawInput());
    renderPrediction(initialPrediction);
  })
  .catch(error => {
    console.error(error);
    statusPill.textContent = "Failed to load model.json";
    statusPill.className = "pill negative";
    predictionText.textContent =
      "Check whether model.json is in the same folder and whether the GitHub Pages deployment is correct.";
  });