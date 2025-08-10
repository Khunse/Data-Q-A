import fs from 'fs';
import {QdrantClient} from '@qdrant/js-client-rest'
import { configDotenv } from 'dotenv';
import {InferenceClient} from "@huggingface/inference"
import { randomUUID } from 'crypto';
import {RecursiveCharacterTextSplitter} from '@langchain/textsplitters'
import { PuppeteerWebBaseLoader } from '@langchain/community/document_loaders/web/puppeteer'

configDotenv();


const externalData = [
    'https://www.ayabank.com/digital-services/card-services/credit-card',

]

const splitter = new RecursiveCharacterTextSplitter(
    {
        chunkSize: 512,
        chunkOverlap: 100
    }
)

const datascraping = async(url) => {
    const loader = new PuppeteerWebBaseLoader(url,{
        launchOptions: {
            headless: true,
        },
        gotoOptions: {
            waitUntil: 'domcontentloaded',
        },
        evaluate: async(page, browser) => {
            const content = await page.evaluate(() => document.body.innerHTML);
            await browser.close();
            return content;
        }
    });


    return  (await loader.scrape())?.replace(/<[^>]*>?/g, '');     // replace and remove html tags 
};

var vectordata = [];

try {
    
    await Promise.all(externalData.map(async (url) => {
        const content = await datascraping(url);
        const chunks = await splitter.splitText(content);

       if(chunks.length > 0)
       {
        const client = new InferenceClient( process.env.HUGGINGFACE_API_KEY);

        await Promise.all(chunks.map(async (chunk) => {
            const output = await client.featureExtraction(
                {
                    model: 'intfloat/multilingual-e5-large',
                    inputs: chunk,
                    provider: 'hf-inference',
                }
            );

            // console.log('embedded output:', output);
            vectordata.push({ vector: output, text: chunk });
        }));
       }
       else
       {
           console.log('No chunks found for URL:', url);
       }

    //    console.log('Vector data:', vectordata);

    if(vectordata.length > 0)
    {
        const qclient = new QdrantClient(
            {
                apiKey: process.env.QDRANT_API_KEY,
                url: process.env.QDRANT_URL
            }
        );

        vectordata.forEach((data, indx, arr) => {
            arr[indx] = {
                id: randomUUID(),
                vector: data.vector,
                payload: {
                    title: data.text
                }
            }
        });

        const insetresult = await qclient.upsert('customdatas', {
            points: vectordata
        });

        console.log('insert result:', insetresult);

    }
    else
    {
        console.log('No data found to process.');
    }
}));

} catch (error) {
    console.error('Error during data scraping:', error);
}
// try {
//     const data = await fs.promises.readdir('./data');

//     if(data.length > 0)
//     {

//         let vectordata = [];

//         const client = new InferenceClient( process.env.HUGGINGFACE_API_KEY);

//         await Promise.all(data.map(async afile => {


//             const filecon = await fs.promises.readFile(`./data/${afile}`, 'utf-8');
// //  await Promise.all(data.map(async (filedata) => {
// // const resultfile = await ocr(filecon);
// // const resultfile = await worker.recognize(filecon);
// // console.log('result file:', resultfile.data.text);
// // fs.promises.writeFile(`./data/${afile}_result.txt`, resultfile.data.text, 'utf-8');
// // await worker.terminate();
//         const output = await client.featureExtraction(
//             {
//                 model: 'intfloat/multilingual-e5-large',
//                 inputs: filecon,
//                 provider: 'hf-inference',
//             }
//         );

//         console.log('embedded output:',output );
//         vectordata.push({ vector: output, title: afile });
//     }));

//         // }));
//         console.log(vectordata);


//         if(vectordata.length >0)
//         {

//               const qclient = new QdrantClient(
//             {
//                 apiKey: process.env.QDRANT_API_KEY,
//                 url: process.env.QDRANT_URL
//             }
//         );

//           vectordata.forEach((data,indx,arr) => {
//                 arr[indx]  = {
//                     id: randomUUID(),
//                     vector: data.vector,
//                     payload: {
//                         title: data.title
//                     }
//                 }
//             });

//             const insetresult = await qclient.upsert('customdata', {
//                 points: vectordata
//             });
    
//             console.log('insert result:', insetresult);

//         }
//         else
//         {
//             console.log('No data found to process.');
//         }
//     }
//     else
//     {
//         console.log('No files found in directory.');
//     }

// } catch (error) {
//     console.error('Error reading directory:', error);
// }