export const SPEAKERS = ["Maya Chen", "Theo Park", "Aria Russo", "Sam Okafor", "Noah Vance"];

export const SENTENCES = [
  "The harbor lights blurred into long gold streaks across the water.",
  "She set the cup down and said nothing for almost a minute.",
  "Every model we shipped this quarter beat the last on prosody.",
  "There's a kind of quiet that only arrives well after midnight.",
  "He counted the steps twice, just to be certain of the number.",
  "Rain moved across the valley in slow, deliberate sheets.",
  "We can finetune the decoder without touching the text encoder.",
  "The voice held its warmth even at the very end of the clip.",
  "Nobody expected the second take to be the one that landed.",
  "Turn left at the old mill and follow the road until it ends.",
];

export const PHON = [
  "ðə ˈhɑɹbɚ laɪts blɝd ˈɪntu lɔŋ goʊld stɹiks əˈkɹɔs ðə ˈwɔtɚ",
  "ʃi sɛt ðə kʌp daʊn ænd sɛd ˈnʌθɪŋ fɔɹ ˈɔlmoʊst ə ˈmɪnɪt",
  "ˈɛvɹi ˈmɑdəl wi ʃɪpt ðɪs ˈkwɔɹtɚ bit ðə læst ɑn ˈpɹɑsədi",
  "ðɛɹz ə kaɪnd ʌv ˈkwaɪət ðæt ˈoʊnli əˈɹaɪvz wɛl ˈæftɚ ˈmɪdnaɪt",
  "hi ˈkaʊntəd ðə stɛps twaɪs dʒʌst tu bi ˈsɝtən ʌv ðə ˈnʌmbɚ",
  "ɹeɪn muvd əˈkɹɔs ðə ˈvæli ɪn sloʊ dɪˈlɪbɚət ʃits",
  "wi kæn ˈfaɪnˌtun ðə ˈdiˌkoʊdɚ wɪˈðaʊt ˈtʌtʃɪŋ ðə tɛkst ɪnˈkoʊdɚ",
  "ðə vɔɪs hɛld ɪts wɔɹmθ ˈivɪn æt ðə ˈvɛɹi ɛnd ʌv ðə klɪp",
  "ˈnoʊˌbɑdi ɪkˈspɛktəd ðə ˈsɛkənd teɪk tu bi ðə wʌn ðæt ˈlændəd",
  "tɝn lɛft æt ði oʊld mɪl ænd ˈfɑloʊ ðə ɹoʊd ənˈtɪl ɪt ɛndz",
];

/** Deterministic 0–1 pseudo-random from an integer seed. */
export function rng(seed: number): number {
  const x = Math.sin(seed) * 10000;
  return x - Math.floor(x);
}
