# tfviolence: violence image classification with Tensorflow

# Categories
## Ref 1. categories [refer to Baidu API]
    正常 警察部队 血腥 尸体 爆炸火灾 杀人 暴乱 
    暴恐人物 军事武器 暴恐旗帜 血腥动物或动物尸体 
    车祸 特殊服饰 枪械 弹药 刀具 武装人员

## Ref 2.  
    血腥场景
    特殊着装 - 军警制服，作战服，僧服，法轮功服装，吉里巴甫服等
    特殊标志 - ISIS，藏独，台独，。。。
    枪支器械 - 
    军警徽 - 
## Ref 3. 21CN需求

## My Categories
    正常 [normal]
    血腥 [bloody]
    尸体骸骨 [corpse] - 尸体或部分肢体
    爆炸火灾 [bomb] - 放火，爆炸
    暴乱游行 [riot] - 
        - 服装类
    军警部队 [army_police]- 警察部队服装
    特殊服饰 [army_terror]- 明显的暴恐服装-僧服，法轮功服装，吉里巴甫服等
    武装人员 [army_other]- 除军警部队和特殊服装之外
        - 标志类
    军警标志 [sign_normal] - 国旗，国徽，警徽等
    暴恐标志 [sign_terror] - ISIS，台独，藏独
        - 武器类
    军事武器 [weapon_large] - 大型武器，坦克导弹
    枪械弹药 [weapon_small] - 小型武器
    刀具 [weapon_knife] - 冷兵器


# image classification using slim functionalities.
# TODO: use tf.Keras for training and validation
