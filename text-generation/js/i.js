import {fileURLToPath} from "url";
import path from "path";
import {getLlama, LlamaChatSession} from "node-llama-cpp";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const llama = await getLlama();
const model = await llama.loadModel({
    modelPath: path.join(__dirname, "models", "hf_bartowski_gemma-2-27b-it-Q6_K_L.gguf")
});
const context = await model.createContext();
const session = new LlamaChatSession({
    contextSequence: context.getSequence()
});


const q1 = "translate to ru: Hey, buddy! What's up! Did you received my last message?";
console.log("User: " + q1);

const a1 = await session.prompt(q1);
console.log("AI: " + a1);


const q2 = "About ships and vessels. I have json with keys in english and empty values. Fill empty values with translated to ru. Return result as json: { \"Select request type*\": \"\", \"I want to buy\": \"\", \"I want to sell\": \"\", \"I have an open cargo\": \"\", \"I have an open ship\": \"\", \"Select ship type*\": \"\", \"Anchor Handling Tug Supply (AHTS)\": \"\", \"Fast Supply Vessel (FSV)\": \"\", \"Survey\": \"\", \"Work boats\": \"\", \"Tuna Longliners\": \"\", \"Beam Trawler\": \"\", \"Newbuild Vessels\": \"\", \"Title*\": \"\", \"Free-form message\": \"\", \"Type\": \"\", \"Capesize\": \"\", \"sea\": \"\", \"sea-river\": \"\", \"Hull\": \"\", \"DWT\": \"\", \"GRT\": \"\", \"NRT\": \"\", \"LOA\": \"\", \"LBP\": \"\", \"Depth\": \"\", \"Ice class\": \"\", \"Crane cap.\": \"\", \"Crane rev.\": \"\", \"Passengers\": \"\", \"Decks\": \"\", \"Teu\": \"\", \"Cars\": \"\", \"Main Engine\": \"\", \"Type of fuel\": \"\", \"DD last\": \"\", \"DD next\": \"\", \"SS last\": \"\", \"SS next\": \"\", \"Asphalt carrier\": \"\", \"Attach a file\": \"\", \"August\": \"\", \"Australasia\": \"\", \"Auxiliary engine\": \"\", \"Average reefer\": \"\", \"Baltiyskiy\": \"\", \"Barge\": \"\", \"Beam\": \"\", \"Black Sea\": \"\", \"box shaped\": \"\", \"Build year\": \"\", \"Built in\": \"\", \"built year\": \"\", \"Bulk carrier\": \"\", \"Bunkering vessel\": \"\", \"Cable layer\": \"\", \"Capacity\": \"\", \"Car float\": \"\"}";
console.log("User: " + q2);

const a2 = await session.prompt(q2);
console.log("AI: " + a2);

