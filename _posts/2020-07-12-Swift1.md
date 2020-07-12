# Hello

```swift
class EmojiMemoryGame: ObservableObject {
    @Published private var model: MemoryGame<String>
    var theme: MemoryGameTheme = themes.randomElement()!

```
