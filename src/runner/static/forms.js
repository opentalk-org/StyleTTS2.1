function renderSchemaForm(container, schema, values, onChange) {
  container.innerHTML = "";
  const properties = schema.properties || {};
  for (const [name, prop] of Object.entries(properties)) {
    renderSchemaField(container, schema, name, prop, values[name], (next) => {
      values[name] = next;
      onChange(values);
    });
  }
}

function renderSchemaField(container, schema, name, prop, value, onChange) {
  const resolved = resolveSchemaRef(prop, schema);
  const type = schemaType(resolved);
  if (type === "object") {
    renderObjectField(container, schema, name, resolved, value, onChange);
    return;
  }

  const label = document.createElement("label");
  label.className = "field";
  label.textContent = name;
  const input = document.createElement(type === "boolean" || resolved.enum ? "select" : "input");
  if (resolved.enum) input.innerHTML = resolved.enum.map((item) => `<option>${item}</option>`).join("");
  if (type === "boolean") input.innerHTML = "<option>true</option><option>false</option>";
  if (type === "number" || type === "integer") input.type = "number";
  input.value = valueToText(value, resolved);
  input.addEventListener("change", () => onChange(textToValue(input.value, resolved)));
  label.appendChild(input);
  container.appendChild(label);
}

function renderObjectField(container, schema, name, prop, value, onChange) {
  const group = document.createElement("div");
  group.className = "schema-group";
  group.innerHTML = `<p class="schema-title">${name}</p>`;
  const body = document.createElement("div");
  body.className = "schema-fields";
  group.appendChild(body);
  const current = value || {};
  if (prop.properties) renderSchemaForm(body, { ...prop, $defs: schema.$defs }, current, onChange);
  else renderMapField(body, prop, current, onChange);
  container.appendChild(group);
}

function renderMapField(container, prop, value, onChange) {
  const entries = Object.entries(value);
  for (const [key, itemValue] of entries) {
    const row = document.createElement("div");
    row.className = "map-row";
    const keyInput = document.createElement("input");
    keyInput.value = key;
    keyInput.spellcheck = false;
    const valueInput = document.createElement("input");
    valueInput.value = valueToText(itemValue, prop.additionalProperties || {});
    valueInput.type = schemaType(prop.additionalProperties || {}) === "string" ? "text" : "number";
    keyInput.addEventListener("change", () => renameMapKey(value, key, keyInput.value, onChange));
    valueInput.addEventListener("change", () => {
      value[key] = textToValue(valueInput.value, prop.additionalProperties || {});
      onChange(value);
    });
    row.append(keyInput, valueInput, mapRemoveButton(value, key, onChange));
    container.appendChild(row);
  }

  const button = document.createElement("button");
  button.type = "button";
  button.className = "btn btn-ghost btn-block";
  button.textContent = "Add entry";
  button.addEventListener("click", () => {
    value[nextMapKey(value)] = defaultValue(prop.additionalProperties || {});
    onChange(value);
  });
  container.appendChild(button);
}

function mapRemoveButton(value, key, onChange) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "btn btn-ghost";
  button.textContent = "Remove";
  button.addEventListener("click", () => {
    delete value[key];
    onChange(value);
  });
  return button;
}

function renameMapKey(value, oldKey, nextKey, onChange) {
  if (!nextKey || nextKey === oldKey) return;
  value[nextKey] = value[oldKey];
  delete value[oldKey];
  onChange(value);
}

function nextMapKey(value) {
  let index = 1;
  while (`entry_${index}` in value) index += 1;
  return `entry_${index}`;
}

function resolveSchemaRef(prop, schema) {
  if (!prop.$ref) return prop;
  const name = prop.$ref.replace("#/$defs/", "");
  return schema.$defs[name];
}

function valueToText(value, prop) {
  const type = schemaType(prop);
  if (value === null || value === undefined) return "";
  if (type === "array") return value.join(", ");
  return String(value);
}

function textToValue(value, prop) {
  const type = schemaType(prop);
  if (value === "" && (prop.anyOf || []).some((item) => item.type === "null")) return null;
  if (value === "" && !["string", "array"].includes(type)) return null;
  if (type === "integer") return Number.parseInt(value, 10);
  if (type === "number") return Number.parseFloat(value);
  if (type === "boolean") return value === "true";
  if (type === "array") return value.split(",").map((item) => item.trim()).filter(Boolean);
  return value;
}

function defaultValue(prop) {
  const type = schemaType(prop);
  if (type === "integer" || type === "number") return 0;
  if (type === "boolean") return false;
  if (type === "array") return [];
  if (type === "object") return {};
  return "";
}
