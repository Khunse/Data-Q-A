import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf";
import { configDotenv } from "dotenv";
import { QdrantClient } from "@qdrant/js-client-rest";
import {OpenAI} from 'openai'

configDotenv();

const embeddings = new HuggingFaceInferenceEmbeddings({
  apiKey: process.env.HUGGINGFACE_API_KEY,
  model: 'intfloat/multilingual-e5-large',
  provider: 'hf-inference',
});

const qclient = new QdrantClient(
            {
                apiKey: process.env.QDRANT_API_KEY,
                url: process.env.QDRANT_URL
            }
        );

const input = `how can i change my email address that is used for my credit card?`;
let dataContext;

const result = await embeddings.embedQuery(input);

console.log('Embedding result:', result);
const query = await qclient.query('customdatas', {
    with_payload: true,
    with_vector: true,
    vector: result,
    limit: 10
});

if(query?.points.length > 0)
{
    const points = query.points.map(point => ({
        id: point.id,
        vector: point.vector,
        payload: point.payload
    }));

    console.log('Query points:', points);

    dataContext = points.map(point => `- ${point.payload.title}`).join('\n');
}
else
{
    console.log('No results found.');
}

const promptTemplate = {
    role: 'system',
    content : `
You are a helpful assistant. You know everything about AYA Bank which is from Myanmar.
Use the below context to augment what you know about AYA Bank.
The context will provide you with the most recent page data from AYA Bank official website
If the context doesn't include the information you need answer based on your existing knowledge and don't mention the source of your information or what the context does or doesn't include.
Format responses using markdown where applicable and don't return images.

Context:
${dataContext}

User question: ${input}
`
};

try {
    
    const openaiClient = new OpenAI({
        baseURL: 'https://router.huggingface.co/v1',
        apiKey: process.env.HUGGINGFACE_API_KEY
    });

    const chatCompletion = await openaiClient.chat.completions.create({
        model: 'openai/gpt-oss-20b:fireworks-ai',
        messages:[promptTemplate],
    });

    console.log('Chat completion result:', chatCompletion.choices[0].message.content);

} catch (error) {
    console.error('Error during prompt generation:', error);
}