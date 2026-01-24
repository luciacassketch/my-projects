import { assign, createActor, setup } from "xstate";
import { Settings, speechstate } from "speechstate";
import { createBrowserInspector } from "@statelyai/inspect";
import { KEY } from "./Code/src/azure";
import { DMContext, DMEvents } from "./types";

const inspector = createBrowserInspector();


const azureCredentials = {
    endpoint:
      "https://swedencentral.api.cognitive.microsoft.com/sts/v1.0/issuetoken",
    key: KEY,
  };
  const settings: Settings = {
    azureCredentials: azureCredentials,
    azureRegion: "swedencentral",
    asrDefaultCompleteTimeout: 0,
    asrDefaultNoInputTimeout: 5000,
    locale: "en-US",
    ttsDefaultVoice: "en-GB-RyanNeural",
  };


const arenaMap: { [index: string]: {[index: string]: number} } = {
  air_vs_water: {
    blizzard : 4,
    lightning : 3,
    fog : 1,
    whirlwind : 5,
    frost : 2,
    hurricane : 6,
    "tsunami wave" : 6,
    ice : 2,
    "breathless bubble" : 1,
    "rising tide" : 3,
    "whale's swallow" : 4,
    "whirlpool" : 5
  },
  fire_vs_earth: {
    "lava flow" : 4,
    "desert's heat" : 2,
    "solar storm" : 5,
    "spewing flame" : 3,
    "burning coal" : 1,
    "volcano storm" : 6,
    earthquake : 6,
    fossil : 4,
    landslide : 5,
    "tree's squeeze" : 2,
    "dust storm" : 1,
    "stone storm" : 3
  },
  air_vs_fire: {
    blizzard : 3,
    lightning : 2,
    fog : 1,
    whirlwind : 5,
    frost : 4,
    hurricane : 6,
    "lava flow" : 5,
    "desert's heat" : 2,
    "solar storm" : 6,
    "spewing flame" : 3,
    "burning coal" : 1,
    "volcano storm" : 4
  },
  air_vs_earth: {
    blizzard : 4,
    lightning : 2,
    fog : 1,
    whirlwind : 5,
    frost : 3,
    hurricane : 6,
    earthquake : 6,
    fossil : 4,
    landslide : 5,
    "tree's squeeze" : 2,
    "dust storm" : 1,
    "stone storm" : 3
  },
  fire_vs_water: {
    "lava flow" : 5,
    "desert's heat" : 3,
    "solar storm" : 6,
    "spewing flame" : 1,
    "burning coal" : 2,
    "volcano storm" : 4,
    "tsunami wave" : 6,
    ice : 4,
    "breathless bubble" : 2,
    "rising tide" : 3,
    "whale's swallow" : 1,
    "whirlpool" : 5
  },
  water_vs_earth: {
    "tsunami wave" : 6,
    ice : 3,
    "breathless bubble" : 2,
    "rising tide" : 1,
    "whale's swallow" : 4,
    "whirlpool" : 5,
    earthquake : 6,
    fossil : 2,
    landslide : 5,
    "tree's squeeze" : 1,
    "dust storm" : 3,
    "stone storm" : 4
  }
}

const characters: {[index: string] : string[]} = {
  air : ["rage","danger","bolt"],
  earth : ["soul crusher","spite","stoneheart"],
  fire : ["madness","trigger","power"],
  water : ["bloodlust","thirst","nightmare"]
}

const elements: string[] = ["air", "fire", "water", "earth"]

const elementsMap: { [key: string]: string[] } = {
  air: ["blizzard", "lightning", "fog", "whirlwind", "frost", "hurricane"],
  fire: ["lava flow", "desert's heat", "solar storm", "spewing flame", "burning coal", "volcano storm"],
  water: ["tsunami wave", "ice", "breathless bubble", "rising tide", "whale's swallow", "whirlpool"],
  earth: ["earthquake", "fossil", "landslide", "tree's squeeze", "dust storm", "stone storm"]
};


const booster: {[index: string] : string} = {
  rage: "frost", //fire, maybe earth
  danger: "blizzard", //pretty much all of them
  bolt: "lightning", //water
  "soul crusher": "fossil", //air
  spite: "dust storm", //water
  stoneheart: "stone storm", //pretty much all of them
  madness: "spewing flame", //air and earth
  trigger: "desert's heat", //water
  power: "burning coal", //water (very little)
  bloodlust: "ice", //fire, maybe earth
  thirst: "rising tide", //fire and air
  nightmare: "whale's swallow"//water
}

function yes(utterance: string) {
  return (utterance.toLowerCase().includes("yes") || utterance.toLowerCase().includes("yeah") );
}
function no(utterance: string) {
  return (utterance.toLowerCase().includes("no") || utterance.toLowerCase().includes("nope") );
}

function repeat(utterance: string) {
  return (utterance.toLowerCase().includes("repeat") || utterance.toLowerCase().includes("again") );
}
function repair(utterance: string) {
  return (utterance.toLowerCase().includes("sorry") || utterance.toLowerCase().includes("wait") );
}
function exit(utterance: string) {
  return (utterance.toLowerCase().includes("exit"));
}
function reset(utterance: string) {
  return (utterance.toLowerCase().includes("reset"));
}

function random(n: number) {
  let i: number = Math.floor(Math.random() * n);
  return i;
}
function randomElemAndChar() {
  let i: number = random(4);
  console.log("random:",i)
  let elem: string =  elements[i];
  let j: number = random(3);
  let char: string = characters[elem][j];
  return [elem, char];
}

function randomMove(element: string, pcelement: string, arena: string, char: string, pcchar: string, boost: boolean = false) {
  let i: number = random(6);
  let j: number = random(6)
  let move = elementsMap[element][i];
  let pcmove = elementsMap[pcelement][j];
  let point: number = arenaMap[arena][move];
  let pcpoint: number = arenaMap[arena][pcmove];
  let winner: string;
  if (boost) {
    if (move == booster[char]) {
      point = point + 1;
    }
    if (pcmove == booster[pcchar]) {
      pcpoint = pcpoint + 1;
    }
  }

  if (point > pcpoint) {
    winner = char;
  }
  else if (pcpoint > point) {
    winner = pcchar;
  }
  else {
    winner = "both"
  }
  return [move, pcmove, winner];
}

function returnArena(elem: string, pcelem: string) {
  let possibilities: string[] = ["air_vs_water", "fire_vs_earth", "air_vs_fire", "air_vs_earth", "fire_vs_water", "water_vs_earth"];
  let arena: string;
  if (elem != pcelem) {
    arena = elem.concat("_vs_", pcelem);
    if (!possibilities.includes(arena)) {
      arena = pcelem.concat("_vs_", elem)
    }
  }
  else {
    let i: number = random(3);
    const arrayWithoutElem = elements.filter(function (e) {
      return e !== elem;
    });
    let other_elem: string =  arrayWithoutElem[i];
    arena = elem.concat("_vs_", other_elem);
    if (!possibilities.includes(arena)) {
      arena = other_elem.concat("_vs_", elem)
    }
  }
  return arena;
}


const dmMachine = setup({
  types: {
    context: {} as DMContext,
    events: {} as DMEvents,
  },
  actions: {
    "spst.speak": ({ context }, params: { utterance: string }) =>
      context.spstRef.send({
        type: "SPEAK",
        value: {
          utterance: params.utterance,
        },
      }),
    "spst.listen": ({ context }) =>
      context.spstRef.send({
        type: "LISTEN",
      }),
  },
}).createMachine({
  context: ({ spawn }) => ({
    spstRef: spawn(speechstate, { input: settings }),
    lastResult: null, 
    elem: null,
    char: null,
    pcranchoice: null,
    battle: null,
    mypoint: 0,
    pcpoint: 0,
    secondRound: false,
    mySecondpoint: 0,
    pcSecondpoint: 0
  }),
  id: "DM",
  initial: "Prepare",
  states: {
    Prepare: {
      entry: ({ context }) => context.spstRef.send({ type: "PREPARE" }),
      on: { ASRTTS_READY: "WaitToStart" },
    },
    WaitToStart: {
      on: { CLICK: "SkipIntro" },
    },
    SkipIntro: {
      initial : "Prompt",
      on : { 
        LISTEN_COMPLETE: [
        {
          target: "Exit",
          guard: ({ context }) => !!context.lastResult && exit(context.lastResult![0]?.utterance),
        },
        {
          target: "Reset",
          guard: ({ context }) => !!context.lastResult && reset(context.lastResult![0]?.utterance),
        },
        {
          target: "PcRandom",
          guard: ({ context }) => !!context.lastResult && yes(context.lastResult![0]?.utterance),
        },
        {
          target: "Greeting",
          guard: ({ context }) => !!context.lastResult && no(context.lastResult![0]?.utterance),
        },
        {
          target: ".Repair",
          guard: ({ context }) => !!context.lastResult && repair(context.lastResult![0]?.utterance),
        },
        { target: ".NoInput" },
      ]
     },
     states: {
      Prompt: {
        entry: { type: "spst.speak", 
          params: { utterance: "Welcome to battle of the elements! If you already know how to play,\
            you can skip this introduction. You can also use all of the other shortcuts\
            to exit or reset the game. Do you want to skip the introduction?" } },
        on: { SPEAK_COMPLETE: "Listen" },
      },
      Listen: {
        entry: { type: "spst.listen" },
        on: {
          RECOGNISED: {
            actions: assign(({ event }) => {
              return { lastResult: event.value }
            }),
          },
          ASR_NOINPUT: { actions: assign({ lastResult: null }) },
        },
      },
      NoInput: {
        entry: { type: "spst.speak", params: { utterance: "I didn't catch that, please try again." } },
        on: { SPEAK_COMPLETE: "Listen" },
      },
      Repair: {
        entry: { type: "spst.speak", params: { utterance: "I heard you're unsure, what is your final choice?" } },
        on: { SPEAK_COMPLETE: "Listen" },
      }
    }
    },
    Greeting: {
      entry: { type: "spst.speak", 
        params: { utterance: "Welcome to battle of the elements! To exit the game, say that you want to exit,\
          or just say exit. To restart the game without hearing this introduction, ask to reset,\
          or simply say reset. If you need me to repeat a question during the game, just ask. \
          To play, you need to choose your element, amongst \
          air, fire, water and earth. Then, you will choose your player. \
          The game will have two rounds, and in both of them your moves will be randomly \
          selected by the computer. But the character you choose matters, because in the second round \
          they can boost specific moves, which will give you advantages or disadvantages \
          depending on the element you fight against. So choose wisely. Whoever wins the second\
          round wins the game." } },
      on: { SPEAK_COMPLETE: "PcRandom" },
    },
    PcRandom: {
      entry: { 
        type: "spst.speak", 
        params: { utterance: `Now the computer will choose its element and a character`},
      },  
      on: { 
        SPEAK_COMPLETE: [
        {
          target: "ChooseElement",
          actions: assign({ pcranchoice: () => randomElemAndChar() }),
        }
      ]
    },
    },
    ChooseElement: {
      initial: "Prompt",
      on: {
        LISTEN_COMPLETE: [
          {
            target: "Exit",
            guard: ({ context }) => !!context.lastResult && exit(context.lastResult![0]?.utterance),
          },
          {
            target: "Reset",
            guard: ({ context }) => !!context.lastResult && reset(context.lastResult![0]?.utterance),
          },
          {
            target: "Air",
            guard: ({ context }) => !!context.lastResult && context.lastResult![0]?.utterance.toLowerCase() == "air",
            actions: assign({ elem: "air" })
          },
          {
            target: "Fire",
            guard: ({ context }) => !!context.lastResult && context.lastResult![0]?.utterance.toLowerCase() == "fire",
            actions: assign({ elem: "fire" })
          },
          {
            target: "Earth",
            guard: ({ context }) => !!context.lastResult && context.lastResult![0]?.utterance.toLowerCase() == "earth",
            actions: assign({ elem: "earth" })
          },
          {
            target: "Water",
            guard: ({ context }) => !!context.lastResult && context.lastResult![0]?.utterance.toLowerCase() == "water",
            actions: assign({ elem: "water" })
          },
          {
            target: ".Prompt",
            guard: ({ context }) => !!context.lastResult && repeat(context.lastResult![0]?.utterance)
          },
          {
            target: ".Repair",
            guard: ({ context }) => !!context.lastResult && repair(context.lastResult![0]?.utterance)
          },
          { target: ".NoInput" },
        ],
      },
      states: {
        Prompt: {
          entry: { 
            type: "spst.speak", 
            params: ({ context }) => ({
              utterance: `The computer will fight with ${context.pcranchoice![0]}\
              and its character will be ${context.pcranchoice![1]}. Now, what element do you want to play with? `
            })
          },  
          on: {SPEAK_COMPLETE: "Listen"}
        },
        Listen: {
          entry: { type: "spst.listen" },
          on: {
            RECOGNISED: {
              actions: assign(({ event }) => {
                return { lastResult: event.value }
              }),
            },
            ASR_NOINPUT: { actions: assign({ lastResult: null }) },
          },
        },
        NoInput: {
          entry: { type: "spst.speak", params: { utterance: "I didn't catch that, please try again." } },
          on: { SPEAK_COMPLETE: "Listen" },
        },
        Repair: {
          entry: { type: "spst.speak", params: { utterance: "I heard you're unsure, what is your final choice?" } },
          on: { SPEAK_COMPLETE: "Listen" },
        }
      },
    },
    Air: {
      initial : "Prompt",
      on : { 
        LISTEN_COMPLETE: [
        {
          target: "Exit",
          guard: ({ context }) => !!context.lastResult && exit(context.lastResult![0]?.utterance),
        },
        {
          target: "Reset",
          guard: ({ context }) => !!context.lastResult && reset(context.lastResult![0]?.utterance),
        },
        {
          target: "Battle",
          guard: ({ context }) => !!context.lastResult && characters["air"].includes(context.lastResult![0]?.utterance.toLowerCase()),
          actions: assign(({ context }) => {
            return { char: context.lastResult![0]?.utterance.toLowerCase() }
          }),
        },
        {
          target: ".Prompt",
          guard: ({ context }) => !!context.lastResult && repeat(context.lastResult![0]?.utterance)
        },
        {
          target: ".Repair",
          guard: ({ context }) => !!context.lastResult && repair(context.lastResult![0]?.utterance)
        },
        { target: ".NoInput" },
      ]
     },
     states: {
      Prompt: {
        entry: { type: "spst.speak", 
          params: { utterance: "As an air warrior, you can choose one of these characters...\
            Danger, a griffin with a sharp beak and deadly claws...\
            Bolt, a lethally quick ghost made of air...\
            Rage, a giant vulture with fury in its eyes.\
            What is your choice?" } },
        on: { SPEAK_COMPLETE: "Listen" },
      },
      Listen: {
        entry: { type: "spst.listen" },
        on: {
          RECOGNISED: {
            actions: assign(({ event }) => {
              return { lastResult: event.value }
            }),
          },
          ASR_NOINPUT: { actions: assign({ lastResult: null }) },
        },
      },
      NoInput: {
        entry: { type: "spst.speak", params: { utterance: "I didn't catch that, please try again." } },
        on: { SPEAK_COMPLETE: "Listen" },
      },
      Repair: {
        entry: { type: "spst.speak", params: { utterance: "I heard you're unsure, what is your final choice?" } },
        on: { SPEAK_COMPLETE: "Listen" },
      }
    }
    },
    Fire: {
      initial : "Prompt",
      on : { 
        LISTEN_COMPLETE: [
        {
          target: "Exit",
          guard: ({ context }) => !!context.lastResult && exit(context.lastResult![0]?.utterance),
        },
        {
          target: "Reset",
          guard: ({ context }) => !!context.lastResult && reset(context.lastResult![0]?.utterance),
        },
        {
          target: "Battle",
          guard: ({ context }) => !!context.lastResult && characters["fire"].includes(context.lastResult![0]?.utterance.toLowerCase()),
          actions: assign(({ context }) => {
            return { char: context.lastResult![0]?.utterance.toLowerCase() }
          }),
        },
        {
          target: ".Prompt",
          guard: ({ context }) => !!context.lastResult && repeat(context.lastResult![0]?.utterance)
        },
        {
          target: ".Repair",
          guard: ({ context }) => !!context.lastResult && repair(context.lastResult![0]?.utterance)
        },
        { target: ".NoInput" },
      ]
     },
     states: {
      Prompt: {
        entry: { type: "spst.speak", 
          params: { utterance: "As a fire warrior, you can choose one of these characters...\
            Madness, a giant lion with flames as mane...\
            Trigger, a sneaky monster, only waiting to be pushed to attack...\
            Power, a deadly dragon with eyes as burning coal.\
            What is your choice?" } },
        on: { SPEAK_COMPLETE: "Listen" },
      },
      Listen: {
        entry: { type: "spst.listen" },
        on: {
          RECOGNISED: {
            actions: assign(({ event }) => {
              return { lastResult: event.value }
            }),
          },
          ASR_NOINPUT: { actions: assign({ lastResult: null }) },
        },
      },
      NoInput: {
        entry: { type: "spst.speak", params: { utterance: "I didn't catch that, please try again." } },
        on: { SPEAK_COMPLETE: "Listen" },
      },
      Repair: {
        entry: { type: "spst.speak", params: { utterance: "I heard you're unsure, what is your final choice?" } },
        on: { SPEAK_COMPLETE: "Listen" },
      }
    }
    },
    Earth: {
      initial : "Prompt",
      on : { 
        LISTEN_COMPLETE: [
        {
          target: "Exit",
          guard: ({ context }) => !!context.lastResult && exit(context.lastResult![0]?.utterance),
        },
        {
          target: "Reset",
          guard: ({ context }) => !!context.lastResult && reset(context.lastResult![0]?.utterance),
        },
        {
          target: "Battle",
          guard: ({ context }) => !!context.lastResult && characters["earth"].includes(context.lastResult![0]?.utterance.toLowerCase()),
          actions: assign(({ context }) => {
            return { char: context.lastResult![0]?.utterance.toLowerCase() }
          }),
        },
        {
          target: ".Prompt",
          guard: ({ context }) => !!context.lastResult && repeat(context.lastResult![0]?.utterance)
        },
        {
          target: ".Repair",
          guard: ({ context }) => !!context.lastResult && repair(context.lastResult![0]?.utterance)
        },
        { target: ".NoInput" },
      ]
     },
     states: {
      Prompt: {
        entry: { type: "spst.speak", 
          params: { utterance: "As an earth warrior, you can choose one of these characters...\
            Soul Crusher, a human-looking tree, with long fingers to squeeze its opponent...\
            Spite, an over-sized warrior-ant, eager to attack...\
            Stoneheart, a creepy gargoyle, made of indestructible stone.\
            What is your choice?" } },
        on: { SPEAK_COMPLETE: "Listen" },
      },
      Listen: {
        entry: { type: "spst.listen" },
        on: {
          RECOGNISED: {
            actions: assign(({ event }) => {
              return { lastResult: event.value }
            }),
          },
          ASR_NOINPUT: { actions: assign({ lastResult: null }) },
        },
      },
      NoInput: {
        entry: { type: "spst.speak", params: { utterance: "I didn't catch that, please try again." } },
        on: { SPEAK_COMPLETE: "Listen" },
      },
      Repair: {
        entry: { type: "spst.speak", params: { utterance: "I heard you're unsure, what is your final choice?" } },
        on: { SPEAK_COMPLETE: "Listen" },
      }
    }
    },
    Water: {
      initial : "Prompt",
      on : { 
        LISTEN_COMPLETE: [
        {
          target: "Exit",
          guard: ({ context }) => !!context.lastResult && exit(context.lastResult![0]?.utterance),
        },
        {
          target: "Reset",
          guard: ({ context }) => !!context.lastResult && reset(context.lastResult![0]?.utterance),
        },
        {
          target: "Battle",
          guard: ({ context }) => !!context.lastResult && characters["water"].includes(context.lastResult![0]?.utterance.toLowerCase()),
          actions: assign(({ context }) => {
            return { char: context.lastResult![0]?.utterance.toLowerCase() }
          }),
        },
        {
          target: ".Prompt",
          guard: ({ context }) => !!context.lastResult && repeat(context.lastResult![0]?.utterance)
        },
        {
          target: ".Repair",
          guard: ({ context }) => !!context.lastResult && repair(context.lastResult![0]?.utterance)
        },
        { target: ".NoInput" },
      ]
     },
     states: {
      Prompt: {
        entry: { type: "spst.speak", 
          params: { utterance: "As a water warrior, you can choose one of these characters...\
            Bloodlust, a deadly siren with shark teeth and fins...\
            Thirst, a scary and human-sized slug, equipped to suck blood...\
            Nightmare, a mysterious creature, emerging from the darkest abyss.\
            What is your choice?" } },
        on: { SPEAK_COMPLETE: "Listen" },
      },
      Listen: {
        entry: { type: "spst.listen" },
        on: {
          RECOGNISED: {
            actions: assign(({ event }) => {
              return { lastResult: event.value }
            }),
          },
          ASR_NOINPUT: { actions: assign({ lastResult: null }) },
        },
      },
      NoInput: {
        entry: { type: "spst.speak", params: { utterance: "I didn't catch that, please try again." } },
        on: { SPEAK_COMPLETE: "Listen" },
      },
      Repair: {
        entry: { type: "spst.speak", params: { utterance: "I heard you're unsure, what is your final choice?" } },
        on: { SPEAK_COMPLETE: "Listen" },
      }
    }
    },
    Battle: {
      entry: { 
          type: "spst.speak", 
          params: ({ context }) => ({
            utterance: `Let's fight! ${context.char} versus ${context.pcranchoice![1]}. `,
          })
        },
        on: { 
          SPEAK_COMPLETE: [
            {
              guard: ({ context }) => !context.secondRound,
              actions: assign(({ context }) => {
                return { battle: randomMove(context.elem!, context.pcranchoice![0], returnArena(context.elem!, context.pcranchoice![0]), context.char!, context.pcranchoice![1]),
                 }
              }),
              target: "CheckResult"
            },
            {
              guard: ({ context }) => context.secondRound,
              actions: assign(({ context }) => {
                return { battle: randomMove(context.elem!, context.pcranchoice![0], returnArena(context.elem!, context.pcranchoice![0]), context.char!, context.pcranchoice![1], true),
                 }
              }),
              target: "CheckResult"
            }
          ]
        },  
    },
    CheckResult: {
      entry: { 
        type: "spst.speak", 
        params: ({ context }) => ({
          utterance: `Your move, ${context.battle![0]}, against the computer's, ${context.battle![1]}. \
          ${context.battle![2]} won! `,
        })
      },
      on: { SPEAK_COMPLETE: [
        {
          guard: ({ context }) => (context.battle![2] == context.char) && !context.secondRound,
          actions: assign(({ context }) => {
            return { mypoint: context.mypoint + 1 }
          }),
          target:"CheckPoint" 
        },
        {
          guard: ({ context }) => (context.battle![2] == context.pcranchoice![1]) && !context.secondRound,
          actions: assign(({ context }) => {
            return { pcpoint: context.pcpoint + 1 }
          }),
          target:"CheckPoint" 
        },
        { 
          guard: ({ context }) => context.battle![2] == "both",
          target: "Battle"
        },
        {
          guard: ({ context }) => (context.battle![2] == context.char) && context.secondRound,
          actions: assign(({ context }) => {
            return { mySecondpoint: context.mySecondpoint + 1 }
          }),
          target:"CheckPoint" 
        },
        {
          guard: ({ context }) => (context.battle![2] == context.pcranchoice![1]) && context.secondRound,
          actions: assign(({ context }) => {
            return { pcSecondpoint: context.pcSecondpoint + 1 }
          }),
          target:"CheckPoint" 
        },
      ]},    
    },
    CheckPoint: {
      entry: { 
        type: "spst.speak", 
        params: ({ context }) => ({
          utterance: `${context.battle![2]} \
          ${context.secondRound ? `goes up to ${context.battle![2] == context.char ? `${context.mySecondpoint}` : `${context.pcSecondpoint}`!}` : `goes up to ${context.battle![2] == context.char ? `${context.mypoint}` : `${context.pcpoint}`!}`}`,
        })
      },
      on: { SPEAK_COMPLETE: [
        {
          guard: ({ context }) => context.mypoint < 3 && context.pcpoint < 3,
          target:"Battle" 
        },
        {
          guard: ({ context }) => (context.mypoint == 3 || context.pcpoint == 3) && !context.secondRound,
          target:"SecondBattle" 
        },
        {
          guard: ({ context }) => context.mySecondpoint < 3 && context.pcSecondpoint < 3,
          target:"Battle" 
        },
        { target: "Aftermath"}
      ]},
    },
    SecondBattle: {
      entry: { 
        type: "spst.speak", 
        params: ({ context }) => ({
          utterance: `You've now reached the end of the first round. \
          In the second round, the characters will be able to boost one move. Your player ${context.char}\
          can boost ${booster[context.char!]}, and the computer's player ${context.pcranchoice![1]} \
          can boost ${booster[context.pcranchoice![1]]}. Let the second round begin!`,
        })
      },
      on: { SPEAK_COMPLETE: [
        {
          actions: assign({ secondRound: true }),
          target:"Battle" 
        }
      ]
    }
    },
    Aftermath: {
      initial: "Prompt",
      on : { 
        LISTEN_COMPLETE: [
        {
          target: "Exit",
          guard: ({ context }) => !!context.lastResult && exit(context.lastResult![0]?.utterance),
        },
        {
          target: "Reset",
          guard: ({ context }) => !!context.lastResult && reset(context.lastResult![0]?.utterance),
        },
        {
          target: ".Repair",
          guard: ({ context }) => !!context.lastResult && repair(context.lastResult![0]?.utterance)
        },
        {
          target: ".Prompt",
          guard: ({ context }) => !!context.lastResult && repeat(context.lastResult![0]?.utterance)
        },
        { target: ".NoInput" },
      ]
     },
      states: {
        Prompt: {
          entry: { 
            type: "spst.speak", 
            params: ({ context }) => ({
              utterance: `This is the end of round two, and \
              ${context.mySecondpoint > context.pcSecondpoint ? `you won! ` :  `the computer won! `}\
              You have now reached the end of the game. To start another game right away, say reset.\
              To exit the game, say exit.`,
            })
          },
          on: { SPEAK_COMPLETE: "Listen" },
        },
        Listen: {
          entry: { type: "spst.listen" },
          on: {
            RECOGNISED: {
              actions: assign(({ event }) => {
                return { lastResult: event.value }
              }),
            },
            ASR_NOINPUT: { actions: assign({ lastResult: null }) },
          },
        },
        NoInput: {
          entry: { type: "spst.speak", params: { utterance: "I didn't catch that, please try again." } },
          on: { SPEAK_COMPLETE: "Listen" },
        },
        Repair: {
          entry: { type: "spst.speak", params: { utterance: "I heard you're unsure, what is your final choice?" } },
          on: { SPEAK_COMPLETE: "Listen" },
        }
      }
    },
    Reset: {
      entry: { type: "spst.speak", params: { utterance: "Resetting the game." } },
      on: { SPEAK_COMPLETE: [
        {
        actions: assign({ pcranchoice: null, secondRound: false, mypoint: 0, pcpoint: 0, mySecondpoint: 0, pcSecondpoint: 0 }),
        target:"PcRandom" 
        }
      ]},
    },
    Exit: {
      entry: { type: "spst.speak", params: { utterance: "If that's what you want, bye!" } },
      on: { SPEAK_COMPLETE: [
        {
        actions: assign({ pcranchoice: null, secondRound: false, mypoint: 0, pcpoint: 0, mySecondpoint: 0, pcSecondpoint: 0 }),
        target:"Done" 
        }
      ]},
    },
    Done: {
      on: {
        CLICK: "SkipIntro",
      },
    },
  }
})

const dmActor = createActor(dmMachine, {
  inspect: inspector.inspect,
}).start();

dmActor.subscribe((state) => {
  console.group("State update");
  console.log("State value:", state.value);
  console.log("State context:", state.context);
  console.groupEnd();
});

export function setupButton(element: HTMLButtonElement) {
  element.addEventListener("click", () => {
    dmActor.send({ type: "CLICK" });
  });
  dmActor.subscribe((snapshot) => {
    const meta: { view?: string } = Object.values(
      snapshot.context.spstRef.getSnapshot().getMeta(),
    )[0] || {
      view: undefined,
    };
    element.innerHTML = `${meta.view}`;
  });
}