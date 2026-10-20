import fs from "node:fs";
const path = new URL("../web/app.js", import.meta.url);
let source = fs.readFileSync(path, "utf8");
source = source.replace('readout_error:Number($("#readout").value)}', 'readout_error:Number($("#readout").value),training_pixels:Number($("#training-pixels").value),score_mode:$("#score-mode").value}');
source = source.replace('synthetic instructions`)', 'template-held-out instructions`)');
source = source.replace('if(installedStatus.public_demo){$("#dataset").disabled=true;', 'if(installedStatus.public_demo){$("#bits").querySelector(\'option[value="5"]\')?.setAttribute("disabled","");$("#steps").max="100";$("#dataset").disabled=true;');
source = source.replace('Seed ${result.config.seed}. ${result.limits[0]}', 'Seed ${result.config.seed} · ${result.config.score_mode || "absolute"} residual. ${result.limits[0]}');
fs.writeFileSync(path, source);
