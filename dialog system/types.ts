import { Hypothesis, SpeechStateExternalEvent } from "speechstate";
import { AnyActorRef } from "xstate";

export interface DMContext {
  spstRef: AnyActorRef;
  lastResult: Hypothesis[] | null;
  elem: string | null,
  char: string | null,
  pcranchoice: string[] | null,
  battle: string[] | null,
  mypoint: number,
  pcpoint: number,
  secondRound: boolean,
  mySecondpoint: number,
  pcSecondpoint: number
}

export type DMEvents = SpeechStateExternalEvent | { type: "CLICK" };
