# PubNub Functions AI

Detect anomalies and other out-of-range data points.
This AI is different.
This AI can learn in realtime.
It remembers what it saw, and makes adjustments to its memory.

### Access Details

 - Channel: `anomaly`
 - PubKey: `pub-c-2ef8bfc4-cd94-42d5-8462-8aff1491c454`
 - SubKey: `sub-c-4ad762be-5e68-11ea-b7ea-f2f107c29c38`

### Sample Message

This is the expected input payload.

```json
{
    "temperature": 39,
    "timestamp": 1416331661
}
```

This is the resulting output payload.

```json
{
    "temperature": 39,
    "timestamp": 1416331661,
    "analysis": {
        "threshold": 5,
        "temperature": 39,
        "date": "Tue Nov 18 2014 17:27:41 GMT+0000 (UTC)",
        "vector": [...],
        "expected": 50.991028027646216,
        "offset": 11.991028027646216,
        "proximity": "76%",
        "safe": false
    }
}
```

### Refernece Links

 - [Function Dashboard](https://admin.pubnub.com/#/user/194894/account/194894/app/231266/key/785554/block/59750/editor/63296)
 - [AI Matrix Pre-trained](https://stephenlb.github.io/pubnub-functions-ai/temperature.pre-trained.json)
 - [2019 Prior Work](https://github.com/stephenlb/pubnub-functions-machine-learning)
