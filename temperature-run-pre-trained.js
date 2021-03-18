// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
// Imports
// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
const ai = require('./ai.js');

// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
// Main
// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

async function main() {
    const testing = generateTestTrainingSampleSet(16);

    // AI Model
    const matrix = await ai.Importer.loadFile('./temperature.pre-trained.json');
    const nn = new ai.NeuralNet({ type : 'tanh', learn : 0.001 });
    nn.load(matrix);

    // need a vectorizor
    // makes sure time +1 so minute/hour/day don't zero out

    // Predict
    let inputs      = testing.map( m => m[0] );
    let targets     = testing.map( m => m[1] );
    let predictions = nn.predict(inputs).map(a => a.map(a => Math.round(a)));

    console.log('\n');
    console.log('predictions', predictions.map(m => m[0]));
    console.log('targets    ', targets.map(m => m[0]));
    console.log('\n');
}

// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
// Generate Test Data
// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
function generateTestTrainingSampleSet(samples) {
    return ai.Tensor(samples, i => generateTestTrainingSample() );
}
    
function generateTestTrainingSample() {
    // Ranges of time
    const days    =  7; // Day of Week 1-7
    const hours   = 24; // Hour
    const minutes = 60; // Minute

    // Bit depths
    const dayBits    = 7; // Day of Week 1-7 OneHot
    const hourBits   = 5; // Hour            Encoded
    const minuteBits = 6; // Minute          Encoded

    // Random Sample Generation
    const day    = rnd(days);
    const hour   = rnd(hours);
    const minute = rnd(minutes);

    // Create Vectorized Sample
    const input = []
        .concat(ai.Tensor(dayBits,    i => day === 1 + i           ? 1 : 0))
        .concat(ai.Tensor(hourBits,   i => Math.pow(2, i) & hour   ? 1 : 0))
        .concat(ai.Tensor(minuteBits, i => Math.pow(2, i) & minute ? 1 : 0));
    const base    = input[0] || input[1] ? 80 : 40;
    const output  = [base - rnd(2) + rnd(2)];

    return [input, output];
}

// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
// Time Vectorizor
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
// Inputs
//     3bit - Day-of-week 1 to 7
//     5bit - Hour 1 to 24
//     6bit - Minute 1 to 60
// 
// Output
//     64bit - Safe Temperature Rating
// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
function timeVector(date) {
    const dayBits     = 7; // Day of Week 1-7 OneHot
    const hourBits    = 5; // Hour            Encoded
    const minuteBits  = 6; // Minute          Encoded

    const day         = date.getDay()     + 1;
    const hour        = date.getHours()   + 1;
    const minute      = date.getMinutes() + 1;

    const matrix      = []
        .concat(ai.Tensor(dayBits,    i => day === 1 + i           ? 1 : 0))
        .concat(ai.Tensor(hourBits,   i => Math.pow(2, i) & hour   ? 1 : 0))
        .concat(ai.Tensor(minuteBits, i => Math.pow(2, i) & minute ? 1 : 0));

    return matrix;
}

// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
// Create Date from Unix Timestamp in Seconds
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
// const mydate = dateFromTimestamp(1616042947);
// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
function dateFromTimestamp(timestamp) {
    return timestamp ? new Date(timestamp * 1000) : new Date();
}

// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
// Simple Random Function
// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
function rnd(r) {
    return Math.ceil(Math.random() * r);
}

// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
// Run Main
// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
main();
